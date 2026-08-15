import torch
import torch.nn as nn
from Encoder import Encoder
from Decoder import Decoder


class Transformer(nn.Module):
    def __init__(self,src_vocab_size,tgt_vocab_size,d_model=512,num_head=8,d_ff=2048,num_layers=6,max_len=5000,dropout=0.1,pad_idx=0,):
        super().__init__()
        self.pad_idx=pad_idx 
        self.encoder=Encoder(src_vocab_size, d_model, num_head, d_ff, num_layers, max_len, dropout)
        self.decoder=Decoder(tgt_vocab_size, d_model, num_head, d_ff, num_layers, max_len, dropout)
        self.generator = nn.Linear(d_model, tgt_vocab_size)
        for p in self.parameters():
            if p.dim()>1:
                nn.init.xavier_uniform_(p)
    def make_src_mask(self, src):
        return (src!=self.pad_idx).unsqueeze(1).unsqueeze(2)
    def make_tgt_mask(self, tgt):
        batch, seq_len = tgt.shape
        pad_mask=(tgt != self.pad_idx).unsqueeze(1).unsqueeze(2) 
        look_ahead=torch.tril(torch.ones(seq_len,seq_len,device=tgt.device)).bool()
        return pad_mask & look_ahead
    def forward(self, src, tgt):
        src_mask=self.make_src_mask(src)
        tgt_mask=self.make_tgt_mask(tgt)
        enc_out=self.encoder(src,src_mask)
        dec_out=self.decoder(tgt,enc_out,src_mask,tgt_mask)
        return self.generator(dec_out)