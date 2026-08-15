import torch
import torch.nn as nn
import math
class MultiHeadAttention(nn.Module):
    def __init__(self,d_model:int,num_heads:int,dropout: float=0.1):
        super().__init__()
        assert d_model%num_heads==0,"model dimension dhould be divisible by the heads"
        self.d_model=d_model
        self.num_heads=num_heads
        self.d_k=d_model//num_heads
        self.w_k=nn.Linear(d_model,d_model)
        self.w_q=nn.Linear(d_model,d_model)
        self.w_v=nn.Linear(d_model,d_model)
        self.w_O=nn.Linear(d_model,d_model)
        self.dropout=nn.Dropout(dropout)
        self.attn_weight=None
    def split_heads(self,x:torch.Tensor)->torch.Tensor:
        batch,seq_length,_=x.shape
        x=x.view(batch,seq_length,self.num_heads, self.d_k)
        return x.transpose(1,2)
    def combine_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch, _, seq_len, _ = x.shape
        x = x.transpose(1, 2).contiguous()
        return x.view(batch, seq_len, self.d_model)
    def forward(self,query,key,value,mask=None):
        q=self.split_heads(self.w_q(query))
        k=self.split_heads(self.w_k(key))
        v=self.split_heads(self.w_v(value))
        if mask is not None  and mask.dim()==3:
            mask.unsqueeze(1)
        out,attn=scaled_dot_product(q,k,v,mask,self.dropout)
        self.attn_weight=attn.detach()
        out=self.combine_heads(out)
        return self.w_O(out)