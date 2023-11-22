from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import Dataset


class BilingualDataset(Dataset):
    def __init__(self, ds, tokenizer_src, tokenizer_target, src_lang, target_lang, seq_length):
        super(BilingualDataset, self).__init__()
        self.ds = ds
        self.seq_length = seq_length
        self.tokenizer_src, self.tokenizer_target = tokenizer_src, tokenizer_target
        self.src_lang, self.target_lang = src_lang, target_lang

        self.sos_token = torch.tensor([tokenizer_src.token_to_id("[SOS]")], dtype=torch.int64)
        self.eos_token = torch.tensor([tokenizer_src.token_to_id("[EOS]")], dtype=torch.int64)
        self.pad_token = torch.tensor([tokenizer_src.token_to_id("[PAD]")], dtype=torch.int64)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, index: Any) -> Any:
        src_target_pair = self.ds[index]
        src_text = src_target_pair['translation'][self.src_lang]
        target_text = src_target_pair['translation'][self.target_lang]

        encode_input_tokens = self.tokenizer_src.encode(src_text).ids
        decode_input_tokens = self.tokenizer_target.encode(target_text).ids

        encoder_num_padding_tokens = self.seq_length - len(encode_input_tokens) - 2
        decoder_num_padding_tokens = self.seq_length - len(decode_input_tokens) - 1
        if encoder_num_padding_tokens < 0 or decoder_num_padding_tokens < 0:
            raise ValueError('Sentence is too long')

        encoder_input = torch.cat([
            self.sos_token, torch.tensor(encode_input_tokens, dtype=torch.int64),
            self.eos_token, torch.tensor([self.pad_token] * encoder_num_padding_tokens, dtype=torch.int64)
        ])
        decoder_input = torch.cat([
            self.sos_token, torch.tensor(decode_input_tokens, dtype=torch.int64),
            torch.tensor([self.pad_token] * decoder_num_padding_tokens, dtype=torch.int64)
        ])
        label = torch.cat([
            torch.tensor(decode_input_tokens, dtype=torch.int64),
            self.eos_token, torch.tensor([self.pad_token] * decoder_num_padding_tokens, dtype=torch.int64)
        ])

        assert encoder_input.size(0) == self.seq_length
        assert decoder_input.size(0) == self.seq_length
        assert label.size(0) == self.seq_length

        return {
            'encoder_input': encoder_input,  # (seq_length)
            'decoder_input': decoder_input,  # (seq_length)
            'encoder_mask': (encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int(),  # (1, 1, seq_length)
            # (1, seq_length) -> (1, seq_length, seq_length)
            'decoder_mask': (decoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int() & causal_mask(decoder_input.size(0)),
            'label': label,
            'src_text': src_text, 'target_text': target_text
        }


def causal_mask(size):
    mask = torch.triu(torch.ones(1, size, size), diagonal=1).type(torch.int)
    return mask == 0
