
import torch
import torch.nn as nn
from MultiHeadAttention import MultiHeadAttention
from FeedForwardNetwork import PositionWiseFeedForwardNetwork
import math
class DecoderLayer(nn.Module):
    def __init__(self, d_model,num_head,d_ff,dropout=0.1):
        super().__init__()
        self.self_attn=MultiHeadAttention(d_model,num_head,dropout)
        self.cross_attn=MultiHeadAttention(d_model,num_head,dropout)
        self.ffn=PositionWiseFeedForwardNetwork(d_model, d_ff,dropout)
        self.norm1=nn.LayerNorm(d_model)
        self.norm2=nn.LayerNorm(d_model)
        self.norm3=nn.LayerNorm(d_model)
        self.dropout1=nn.Dropout(dropout)
        self.dropout2=nn.Dropout(dropout)
        self.dropout3=nn.Dropout(dropout)
    def forward(self,x,enc_out,src_mask=None,tgt_mask=None):
        attn_out = self.self_attn(x,x,x,tgt_mask)          # masked self-attention
        x=self.norm1(x+self.dropout1(attn_out))
        cross_out=self.cross_attn(x,enc_out,enc_out,src_mask)  # query=decoder, key/value=encoder
        x=self.norm2(x+self.dropout2(cross_out))
        ffn_out=self.ffn(x)
        x=self.norm3(x+self.dropout3(ffn_out))
        return x