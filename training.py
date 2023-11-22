# https://huggingface.co/datasets/opus_books/viewer/en-no
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dataset import BilingualDataset, causal_mask
from config import get_weights_file_path, get_config
from transformer_model import build_transformer

from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

import warnings


def get_all_sentences(ds, lang):
    for item in ds:
        yield item['translation'][lang]


def get_or_build_tokenizer(config, ds, lang):
    tokenizer_path = Path(config['tokenizer_file'].format(lang))
    if not Path.exists(tokenizer_path):
        tokenizer = Tokenizer(WordLevel(unk_token='[UNK]'))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency=2)
        tokenizer.train_from_iterator(get_all_sentences(ds, lang), trainer=trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return tokenizer


def get_ds(config):
    ds_raw = load_dataset('opus_books', f'{config["lang_src"]}-{config["lang_target"]}', split='train')
    tokenizer_src = get_or_build_tokenizer(config, ds_raw, config['lang_src'])
    tokenizer_target = get_or_build_tokenizer(config, ds_raw, config['lang_target'])

    # keep 90% for training and 10% for validation
    train_ds_size = int(0.9 * len(ds_raw))
    val_ds_size = len(ds_raw) - train_ds_size
    train_ds_raw, val_ds_raw = random_split(ds_raw, [train_ds_size, val_ds_size])

    train_ds = BilingualDataset(ds=train_ds_raw, tokenizer_src=tokenizer_src, tokenizer_target=tokenizer_target,
                                src_lang=config['lang_src'], target_lang=config['lang_target'], seq_length=config['seq_length'])
    val_ds = BilingualDataset(ds=val_ds_raw, tokenizer_src=tokenizer_src, tokenizer_target=tokenizer_target,
                              src_lang=config['lang_src'], target_lang=config['lang_target'], seq_length=config['seq_length'])

    max_len_src, max_len_target = 0, 0
    for item in ds_raw:
        src_ids = tokenizer_src.encode(item['translation'][config['lang_src']]).ids
        target_ids = tokenizer_target.encode(item['translation'][config['lang_target']]).ids
        max_len_src = max(max_len_src, len(src_ids))
        max_len_target = max(max_len_target, len(target_ids))

    print(f'Max length of source-target sentence: {max_len_src} and {max_len_target}')
    train_dataloader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True)
    val_dataloader = DataLoader(val_ds, batch_size=1, shuffle=True)
    return train_dataloader, val_dataloader, tokenizer_src, tokenizer_target


def get_model(config, vocab_src_length, vocab_target_length):
    model = build_transformer(src_vocab_size=vocab_src_length, target_vocab_size=vocab_target_length,
                              src_seq_length=config['seq_length'], target_seq_length=config['seq_length'], d_model=config['d_model'])
    return model


def train_model(config):
    # define device
    device = torch.device('mps' if (torch.backends.mps.is_built() & torch.backends.mps.is_available()) else 'cpu')
    device = 'cpu'
    print(f'Using device {device}')

    Path(config['model_folder']).mkdir(parents=True, exist_ok=True)
    train_dataloader, val_dataloader, tokenizer_src, tokenizer_target = get_ds(config)

    model = get_model(config, tokenizer_src.get_vocab_size(), tokenizer_target.get_vocab_size()).to(device)
    writer = SummaryWriter(config['experiment_name'])
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'], eps=1e-9)

    initial_epoch = 0
    global_step = 0
    if config['preload']:
        model_filename = get_weights_file_path(config, config['preload'])
        print(f'Preloading model {model_filename}..........')
        state = torch.load(model_filename)
        initial_epoch = state('epoch') + 1
        optimizer.load_state_dict(state['optimizer_state_dict'])
        global_step = state['global_step']

    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer_src.token_to_id('[PAD]'), label_smoothing=0.1).to(device)

    # build training loop
    for epoch in range(initial_epoch, config['num_epochs']):
        model.train()
        batch_iterator = tqdm(train_dataloader, desc=f'Processing epoch {epoch:02d}')
        for batch in batch_iterator:
            encoder_input = batch['encoder_input'].to(device)  # size = (batch, seq_length)
            decoder_input = batch['decoder_input'].to(device)  # size = (batch, seq_length)
            encoder_mask = batch['encoder_mask'].to(device)  # size = (batch, 1, 1, seq_length)
            decoder_mask = batch['decoder_mask'].to(device)  # size = (batch, 1, seq_length, seq_length)

            # run the tensor through the transformer
            encoder_output = model.encode(src=encoder_input, src_mask=encoder_mask)
            decoder_output = model.decode(encoder_output=encoder_output, src_mask=encoder_mask,
                                          target=decoder_input, target_mask=decoder_mask)
            projector_output = model.project(decoder_output)
            label = batch['label'].to(device)

            # calculate loss
            loss = loss_fn(projector_output.view(-1, tokenizer_target.get_vocab_size()), label.view(-1))
            batch_iterator.set_postfix({f'loss': f'{loss.item():6.3f}'})
            writer.add_scalar('train loss', loss.item(), global_step)
            writer.flush()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            global_step += 1

        model_filename = get_weights_file_path(config, f'{epoch:02d}')
        torch.save({
            'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(),
            'global_step': global_step
        }, model_filename)


if __name__ == "__main__":
    warnings.filterwarnings('ignore')
    config = get_config()
    train_model(config)

