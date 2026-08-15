import torch
import torch.nn as nn
from MultiHeadAttention import MultiHeadAttention
from FeedForwardNetwork import PositionWiseFeedForwardNetwork
class EncoderLayer(nn.Module):
    def __init__(self,d_model,num_head,d_ff,dropout=0.1):
        super().__init__()
        self.self_attn=MultiHeadAttention(d_model,num_head,dropout)
        self.ffn=PositionWiseFeedForwardNetwork(d_model,d_ff,dropout)
        self.norm1=nn.LayerNorm(d_model)
        self.norm2=nn.LayerNorm(d_model)
        self.dropout1=nn.Dropout(dropout)
        self.dropout2=nn.Dropout(dropout)
    def forward(self, x, mask=None):
        attn_out = self.self_attn(x,x,x, mask)
        x=self.norm1(x+self.dropout1(attn_out))
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout2(ffn_out))
        return x