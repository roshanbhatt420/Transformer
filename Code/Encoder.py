import torch
import torch.nn as nn
from PositionalEncoding import PositionalEncoding
from EncoderLayer import EncoderLayer
import math
class Encoder(nn.Module):
    def __init__(self,vocab_size,d_model,num_heads,d_ff,num_layers,max_len=5000,dropout=0.1):
        super().__init__()
        self.d_model=d_model
        self.embed=nn.Embedding(vocab_size, d_model)
        self.pos_enc=PositionalEncoding(d_model,max_len,dropout)
        self.layers=nn.ModuleList([EncoderLayer(d_model,num_heads,d_ff,dropout)for _ in range(num_layers)])
        self.norm = nn.LayerNorm(d_model)
    def forward(self,src,src_mask=None):
        x=self.embed(src)*math.sqrt(self.d_model)
        x=self.pos_enc(x)
        for layer in self.layers:
            x = layer(x,src_mask)
        return self.norm(x)