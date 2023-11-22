# %% Importing libs
import math

import torch
import torch.nn as nn


# %% Working with Input Embeddings
class InputEmbeddings(nn.Module):
    def __init__(self, d_model: int, vocab_size: int):
        super(InputEmbeddings, self).__init__()
        self.d_model = d_model  # size of each embedding vector
        self.vocab_size = vocab_size  # size of the dictionary of embeddings
        self.embeddings = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        return self.embeddings(x) * math.sqrt(self.d_model)


# %% Working with Positional Encoding
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, seq_length: int, dropout: float) -> None:
        super(PositionalEncoding, self).__init__()
        self.d_model, self.seq_length = d_model, seq_length
        self.dropout = nn.Dropout(dropout)

        # create a matrix of shape (seq_length, d_model)
        position_encoding = torch.zeros(size=(self.seq_length, self.d_model))

        # create a vector of shape (seq_length, 1)
        position = torch.arange(0, self.seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.d_model, 2).float() * (-math.log(10000.0) / self.d_model))

        # apply the sine and cosine for even and odd positions
        position_encoding[:, 0::2] = torch.sin(position * div_term)
        position_encoding[:, 1::2] = torch.cos(position * div_term)

        position_encoding = position_encoding.unsqueeze(0)  # shape = (1, seq_length, d_model)
        self.register_buffer('position_encoding', position_encoding)

    def forward(self, x):
        x = x + (self.position_encoding[:, : x.shape[1], :]).requires_grad_(False)
        return self.dropout(x)


# %% Working with Layer Normalization
class LayerNormalization(nn.Module):
    def __init__(self, epsilon: float = 10e-6) -> None:
        super(LayerNormalization, self).__init__()
        self.epsilon = epsilon
        self.alpha = nn.Parameter(torch.ones(1))  # multiplied
        self.bias = nn.Parameter(torch.ones(1))  # added

    def forward(self, x: torch.Tensor):
        mean = x.mean(axis=-1, keepdim=True)
        std = x.std(axis=-1, keepdim=True)
        return self.alpha * (x - mean) / (std + self.epsilon) + self.bias


# %% Working with Feed Forward Block
class FeedForwardBlock(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super(FeedForwardBlock, self).__init__()
        self.linear_1 = nn.Linear(d_model, d_ff)  # return parameters for W1 and bias 1
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)  # return parameters for W2 and bias 2

    def forward(self, x):
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))


# %% Working with Multi-Heads Attention Block
class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, d_model: int, heads: int, dropout: float) -> None:
        super(MultiHeadAttentionBlock, self).__init__()
        self.attention_scores = None
        self.d_model, self.heads = d_model, heads
        assert d_model // heads, 'd_model should be divided by head to split into multi-heads'

        self.d_k = d_model // heads  # used to split to multi-heads
        self.w_q = nn.Linear(d_model, d_model)  # weight matrix for query
        self.w_k = nn.Linear(d_model, d_model)  # weight matrix for key
        self.w_v = nn.Linear(d_model, d_model)  # weight matrix for value
        self.w_o = nn.Linear(d_model, d_model)  # output weight matrix
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def cal_attention(query, key, value, mask, dropout: nn.Dropout):
        d_k = query.shape[1]  # take number of heads

        attention_score = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            attention_score.masked_fill_(mask == 0, -1e9)
        attention_score = attention_score.softmax(dim=-1)  # (batch, heads, seq_length, seq_length)

        if dropout is not None:
            attention_score = dropout(attention_score)

        return (attention_score @ value), attention_score

    def forward(self, q, k, v, mask):
        query = self.w_q(q)  # original shape = (batch, seq_length, d_model) -> (batch, seq_length, d_model)
        key = self.w_k(k)  # (batch, seq_length, d_model) -> (batch, seq_length, d_model)
        value = self.w_v(v)  # (batch, seq_length, d_model) -> (batch, seq_length, d_model)

        # reshape into multiple-heads
        # (batch, seq_length, d_models) -> (batch, seq_length, h, d_k) -> (batch, h, seq_length, d_k)
        query = query.view(query.shape[0], query.shape[1], self.heads, self.d_k).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.heads, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.heads, self.d_k).transpose(1, 2)

        # calculate attention score for multi-heads
        x, self.attention_scores = MultiHeadAttentionBlock.cal_attention(query, key, value, mask, self.dropout)
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.heads * self.d_k)

        return self.w_o(x)


# %% Working with Residual Block
class ResidualBlock(nn.Module):
    def __init__(self, dropout: float) -> None:
        super(ResidualBlock, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm_layer = LayerNormalization()

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm_layer(x)))


# %% Working with Encoder Block
class EncoderBlock(nn.Module):
    def __init__(self, self_attention_block: MultiHeadAttentionBlock,
                 feed_forward_block: FeedForwardBlock, dropout: float) -> None:
        super(EncoderBlock, self).__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualBlock(dropout) for _ in range(2)])

    def forward(self, x, src_mask):
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, src_mask))
        x = self.residual_connections[1](x, self.feed_forward_block)
        return x


class Encoder(nn.Module):
    def __init__(self, layers: nn.ModuleList) -> None:
        super(Encoder, self).__init__()
        self.layers = layers
        self.norm_layer = LayerNormalization()

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm_layer(x)


# %% Working with Decoder Block
class DecoderBlock(nn.Module):
    def __init__(self, self_attention_block: MultiHeadAttentionBlock,
                 cross_attention_block: MultiHeadAttentionBlock,
                 feed_forward_block: FeedForwardBlock, dropout: float):
        super(DecoderBlock, self).__init__()
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualBlock(dropout=dropout) for _ in range(3)])

    def forward(self, x, encoder_output, src_mask, target_mask):
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, target_mask))
        x = self.residual_connections[1](x, lambda x: self.cross_attention_block(x, encoder_output, encoder_output,
                                                                                 src_mask))
        x = self.residual_connections[2](x, self.feed_forward_block)
        return x


class Decoder(nn.Module):
    def __init__(self, layers: nn.ModuleList) -> None:
        super(Decoder, self).__init__()
        self.layers = layers
        self.norm_layers = LayerNormalization()

    def forward(self, x, encoder_output, src_mask, target_mask):
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, target_mask)
        return self.norm_layers(x)


# %% Working with Projector block
class ProjectLayer(nn.Module):
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.projector = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        # transform to a tensor with shape = (batch, seq_length, d_model) -> (batch, seq_length, vocab_size)
        return torch.log_softmax(self.projector(x), dim=-1)


class Transformer(nn.Module):
    def __init__(self, encoder: Encoder, decoder: Decoder, src_embed: InputEmbeddings,
                 target_embed: InputEmbeddings, src_position: PositionalEncoding, target_position: PositionalEncoding,
                 projector_layer: ProjectLayer):
        super(Transformer, self).__init__()
        self.encoder, self.decoder = encoder, decoder
        self.src_embed, self.target_embed = src_embed, target_embed
        self.src_pos, self.target_pos = src_position, target_position
        self.projector_layer = projector_layer

    def encode(self, src, src_mask):
        src = self.src_embed(src)
        src = self.src_pos(src)
        return self.encoder(src, src_mask)

    def decode(self, encoder_output, src_mask, target, target_mask):
        target = self.target_embed(target)
        target = self.target_pos(target)
        return self.decoder(target, encoder_output, src_mask, target_mask)

    def project(self, x):
        return self.projector_layer(x)


# %% Working with transformer model
def build_transformer(src_vocab_size: int, target_vocab_size: int, src_seq_length: int,
                      target_seq_length: int, d_model: int = 512, n_blocks: int = 6, heads: int = 8,
                      dropout: float = 0.1, d_ff=2048) -> Transformer:
    src_embedding = InputEmbeddings(d_model, vocab_size=src_vocab_size)
    target_embedding = InputEmbeddings(d_model, vocab_size=target_vocab_size)
    src_position = PositionalEncoding(d_model, seq_length=src_seq_length, dropout=dropout)
    target_position = PositionalEncoding(d_model, seq_length=target_seq_length, dropout=dropout)

    # create encoder blocks
    encoder_blocks = []
    for _ in range(n_blocks):
        encoder_self_attention_block = MultiHeadAttentionBlock(d_model, heads, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        encoder_block = EncoderBlock(self_attention_block=encoder_self_attention_block,
                                     feed_forward_block=feed_forward_block, dropout=dropout)
        encoder_blocks.append(encoder_block)

    # create decoder blocks
    decoder_blocks = []
    for _ in range(n_blocks):
        decoder_self_attention_block = MultiHeadAttentionBlock(d_model, heads, dropout)
        decoder_cross_attention_block = MultiHeadAttentionBlock(d_model, heads, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        decoder_block = DecoderBlock(self_attention_block=decoder_self_attention_block,
                                     cross_attention_block=decoder_cross_attention_block,
                                     feed_forward_block=feed_forward_block, dropout=dropout)
        decoder_blocks.append(decoder_block)

    # create the encoder-decoder
    encoder = Encoder(nn.ModuleList(encoder_blocks))
    decoder = Decoder(nn.ModuleList(decoder_blocks))
    projector_layer = ProjectLayer(d_model, vocab_size=target_vocab_size)

    transformer = Transformer(encoder=encoder, decoder=decoder, src_embed=src_embedding, target_embed=target_embedding,
                              src_position=src_position, target_position=target_position, projector_layer=projector_layer)

    # initialize parameters
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform(p)

    return transformer
