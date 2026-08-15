import math
import torch
import torch.nn as nn
from PositionalEncoding import PositionalEncoding
from DecoderLayer import DecoderLayer

class Decoder(nn.Module):
    def __init__(self, vocab_size,d_model,num_head, d_ff,num_layers,max_len=5000,dropout=0.1):
        super().__init__()
        self.d_model=d_model
        self.embed=nn.Embedding(vocab_size,d_model)
        self.pos_enc=PositionalEncoding(d_model,max_len,dropout)
        self.layers=nn.ModuleList([DecoderLayer(d_model, num_head, d_ff, dropout) for _ in range(num_layers)])
        self.norm=nn.LayerNorm(d_model)
    def forward(self,tgt,enc_out,src_mask=None,tgt_mask=None):
        x=self.embed(tgt)*math.sqrt(self.d_model)
        x=self.pos_enc(x)
        for layer in self.layers:
            x=layer(x,enc_out,src_mask,tgt_mask)
        return self.norm(x)