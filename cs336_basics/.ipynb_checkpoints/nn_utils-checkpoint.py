import torch
import torch.nn as nn
from einops import einsum
import math
import numpy as np


def softmax(x:torch.Tensor,dim:int)->torch.Tensor:
    x_max=torch.max(x,dim=dim,keepdim=True).values
    x_max = torch.where(torch.isfinite(x_max), x_max, torch.zeros_like(x_max))
    x_shifted=x-x_max
    exp_x=torch.exp(x_shifted)
    sum_exp = torch.sum(exp_x, dim=dim, keepdim=True)
    return exp_x / sum_exp.clamp_min(torch.finfo(exp_x.dtype).tiny)



def scaled_dot_product_attention(query, key, value, mask=None):
     d_k = query.size(-1)
     scores = torch.matmul(query, key.transpose(-2, -1))
     scores = scores / math.sqrt(d_k)
     if mask is not None:
         scores = scores.masked_fill(~mask, float('-inf')) 
     attn_weights = softmax(scores, dim=-1)
     output=torch.matmul(attn_weights,value)
     return output




class Linear(nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__()
        # 1. 定义超参数
        self.in_features = in_features
        self.out_features = out_features
        
        # 2. 创建权重矩阵 W，形状必须是 (out_features, in_features)

        self.weight = nn.Parameter(torch.empty((out_features,in_features),device=device,dtype=dtype))
        sigma = math.sqrt(2.0 / (in_features + out_features))
        
        # 3. 截断正态分布初始化
        nn.init.trunc_normal_(self.weight,mean=0,std=sigma,a=-3*sigma, b=3*sigma)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

         return einsum(x, self.weight, "... d_in, d_out d_in -> ... d_out")

        
class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        
        # 1. 创建嵌入矩阵，形状为 (num_embeddings, embedding_dim)
       
        self.weight = nn.Parameter(
            torch.empty((num_embeddings, embedding_dim), device=device, dtype=dtype)
        )
        
        # 2. 截断正态分布初始化
        nn.init.trunc_normal_(self.weight,mean=0,std=1,a=-3, b=3)

    def forward(self, token_ids: torch.LongTensor) -> torch.Tensor:
        # 3. 根据 token_ids 去矩阵里查表
        return self.weight[token_ids]

class Rmsnorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.d_model=d_model
        self.eps=eps
        self.weight = nn.Parameter(
            torch.ones(d_model, device=device, dtype=dtype)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        if in_dtype in (torch.float16, torch.bfloat16):
            x = x.to(torch.float32)
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        x = (x / rms) * self.weight
        return x.to(in_dtype)


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int = None, device=None, dtype=None):
        super().__init__()
        
        if d_ff is None:
            d_ff_calc = int(8 / 3 * d_model)
            # 把它向上取整到 64 的倍数
            d_ff = 64 * math.ceil(d_ff_calc / 64)
            
        self.d_model = d_model
        self.d_ff = d_ff
        
        # 2. 实例化三个 Linear 层
        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)
        self.w3 = Linear(d_model, d_ff, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        w1_out = self.w1(x)
        silu_out = w1_out * torch.sigmoid(w1_out) 
        
        w3_out = self.w3(x)
        
        glu_out = silu_out * w3_out  
        
        return self.w2(glu_out)


