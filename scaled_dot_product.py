import torch 
import math
import torch.nn.functional as F

def scaled_dot_product(q , k, v,mask=None):
    d_k=q.size(-1)
    score=torch.matmul(q,k.transpose(-2,-1))/math.sqrt(d_k)
    if mask is not None:
        score=score.masked_fill(mask==0,float('-inf'))
    attn=F.softmax(score,dim=-1)
    out=torch.matmul(attn,v)
    return  out,attn

q = torch.rand(1, 4, 8)  
k = torch.rand(1, 4, 8)
v = torch.rand(1, 4, 8)

out, attn = scaled_dot_product(q, k, v)
print(out.shape)  
print(attn.shape)   
print(attn.sum(dim=-1)) 
    