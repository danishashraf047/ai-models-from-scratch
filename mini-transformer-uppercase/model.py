import math
import torch
from torch import nn

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=128):
        super().__init__()
        positions = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(positions * div)
        pe[:, 1::2] = torch.cos(positions * div)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        if x.size(1) > self.pe.size(1):
            raise ValueError('Sequence exceeds max_len')
        return x + self.pe[:, :x.size(1)]

class UppercaseTransformer(nn.Module):
    def __init__(self, vocab_size, pad_id=0, d_model=128, nhead=4, layers=2, ff_dim=256, dropout=0.1, max_len=128):
        super().__init__()
        self.pad_id = pad_id
        self.d_model = d_model
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.pos = PositionalEncoding(d_model, max_len)
        self.transformer = nn.Transformer(d_model=d_model, nhead=nhead, num_encoder_layers=layers,
                                          num_decoder_layers=layers, dim_feedforward=ff_dim,
                                          dropout=dropout, batch_first=True)
        # Keep padded tensors dense: the nested-tensor evaluation path is
        # experimental and uses operations that are unavailable on MPS.
        self.transformer.encoder.use_nested_tensor = False
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, src, tgt_in):
        src_pad = src.eq(self.pad_id)
        tgt_pad = tgt_in.eq(self.pad_id)
        tgt_len = tgt_in.size(1)
        causal = torch.triu(torch.ones(tgt_len, tgt_len, device=tgt_in.device, dtype=torch.bool), diagonal=1)
        src_emb = self.pos(self.embed(src) * math.sqrt(self.d_model))
        tgt_emb = self.pos(self.embed(tgt_in) * math.sqrt(self.d_model))
        output = self.transformer(src_emb, tgt_emb, tgt_mask=causal,
                                  src_key_padding_mask=src_pad,
                                  tgt_key_padding_mask=tgt_pad,
                                  memory_key_padding_mask=src_pad)
        return self.head(output)
