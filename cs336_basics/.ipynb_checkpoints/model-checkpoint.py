import torch
import torch.nn as nn
from einops import einsum
import math
import numpy as np
from torch.optim import Optimizer


class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        assert d_k % 2 == 0
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len

        # 1. 计算频率: 1 / theta^(2k/d_k)
        k = torch.arange(0, d_k, 2, device=device).float()
        inv_freq = 1.0 / torch.pow(self.theta, k / d_k)

        # 2. 位置序列
        t = torch.arange(max_seq_len, device=device).float()

        # 3. 外积 -> (max_seq_len, d_k/2)
        freqs = torch.outer(t, inv_freq)

        # 4. 仅保留 d_k/2 的 cos/sin 缓存
        cos_cached = torch.cos(freqs)
        sin_cached = torch.sin(freqs)

        # 5. 注册 buffer (不参与梯度更新)
        self.register_buffer("cos_cached", cos_cached, persistent=False)
        self.register_buffer("sin_cached", sin_cached, persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        # 1. 查缓存，并按位置张量的秩显式对齐到 (..., 1, S, d_k/2)
        cos = self.cos_cached[token_positions]
        sin = self.sin_cached[token_positions]
        if token_positions.dim() == 1:
            cos, sin = cos[None, None], sin[None, None]     # (1, 1, S, d/2)
        else:
            cos, sin = cos[:, None], sin[:, None]            # (B, 1, S, d/2)
    
        # 2. 半精度输入提升到 float32 参与旋转，避免中途降位
        in_dtype = x.dtype
        if in_dtype in (torch.float16, torch.bfloat16):
            x = x.to(torch.float32)
    
        # 3. 相邻成对切分
        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]
    
        out_even = x_even * cos - x_odd * sin
        out_odd = x_even * sin + x_odd * cos
    
        # 4. 交错拼回，替代 empty_like + 两段切片赋值
        out = torch.stack((out_even, out_odd), dim=-1).flatten(-2)
        return out.to(in_dtype)

# class CausalMultiHeadSelfAttention(nn.Module):
#     def __init__(
#         self,
#         d_model: int,
#         num_heads: int,
#         max_seq_len: int | None = None,
#         theta: float | None = None,
#         device=None,
#     ):
#         super().__init__()
#         assert d_model % num_heads == 0, 
#         self.d_model = d_model
#         self.num_heads = num_heads
#         self.d_k = d_model // num_heads
#         self.max_seq_len = max_seq_len

#         # 1. 因果掩码：预建一次，persistent=False 避免污染 state_dict
#         if max_seq_len is not None:
#             self.register_buffer(
#                 "causal_mask",
#                 torch.tril(
#                     torch.ones(max_seq_len, max_seq_len, dtype=torch.bool, device=device)
#                 ),
#                 persistent=False,
#             )
#         else:
#             self.causal_mask = None

#         # 2. RoPE：theta 为 None 就整个关掉
#         self.rope = None
#         if theta is not None:
#             assert max_seq_len is not None, "启用 RoPE 时必须提供 max_seq_len"
#             self.rope = RotaryPositionalEmbedding(
#                 theta=theta, d_k=self.d_k, max_seq_len=max_seq_len, device=device
#             )

#     def forward(
#         self,
#         in_features: torch.Tensor,        # (..., S, d_model)
#         q_proj_weight: torch.Tensor,      # (d_model, d_model)
#         k_proj_weight: torch.Tensor,
#         v_proj_weight: torch.Tensor,
#         o_proj_weight: torch.Tensor,
#         token_positions: torch.Tensor | None = None,   # (S,) 或 (..., S)
#     ) -> torch.Tensor:
#         *batch_dims, S, D = in_features.shape
#         if self.rope is not None:   # 只有启用 RoPE 时才检查
#             assert S <= self.max_seq_len, f"seq_len {S} 超过 max_seq_len {self.max_seq_len}"
#         h, d_k = self.num_heads, self.d_k
#         device = in_features.device

#         if self.causal_mask is not None and S <= self.causal_mask.size(0):
#             mask = self.causal_mask[:S, :S]
#             if mask.device != device:
#                 mask = mask.to(device)
#         else:
#             mask = torch.tril(torch.ones(S, S, dtype=torch.bool, device=device))
            

#         # ---- 投影 ----
#         Q = in_features @ q_proj_weight.T
#         K = in_features @ k_proj_weight.T
#         V = in_features @ v_proj_weight.T

#         # ---- 拆多头 ----
#         Q = Q.view(*batch_dims, S, h, d_k).transpose(-3, -2)
#         K = K.view(*batch_dims, S, h, d_k).transpose(-3, -2)
#         V = V.view(*batch_dims, S, h, d_k).transpose(-3, -2)

#         # ---- 可选 RoPE ----
#         if self.rope is not None:
#             if token_positions is None:
#                 token_positions = torch.arange(S, device=device)
#             Q = self.rope(Q, token_positions)
#             K = self.rope(K, token_positions)

#         # ---- 注意力 + 合并 + 输出投影 ----
#         attn_output = scaled_dot_product_attention(Q, K, V, mask)
#         attn_output = attn_output.transpose(-3, -2).contiguous().view(*batch_dims, S, D)
#         return attn_output @ o_proj_weight.T
class CausalMultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads, max_seq_len, theta, device=None, dtype=None):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.max_seq_len = max_seq_len

        self.q_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.k_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.v_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.output_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        
        self.rope = RotaryPositionalEmbedding(theta, self.d_k, max_seq_len, device=device)
        
        # 因果掩码
        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(max_seq_len, max_seq_len, dtype=torch.bool, device=device)),
            persistent=False,
        )

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor = None) -> torch.Tensor:
        *batch_dims, S, _ = x.shape
        h, d_k = self.num_heads, self.d_k
        
        if token_positions is None:
            token_positions = torch.arange(S, device=x.device)
            
        # 投影
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)
        
        # 拆多头
        Q = Q.view(*batch_dims, S, h, d_k).transpose(-3, -2)
        K = K.view(*batch_dims, S, h, d_k).transpose(-3, -2)
        V = V.view(*batch_dims, S, h, d_k).transpose(-3, -2)
        
        # 应用 RoPE
        Q = self.rope(Q, token_positions)
        K = self.rope(K, token_positions)
        
        mask = self.causal_mask[:S, :S]
        
        attn_output = scaled_dot_product_attention(Q, K, V, mask)
        
        # 合并多头
        attn_output = attn_output.transpose(-3, -2).contiguous().view(*batch_dims, S, self.d_model)
        
        return self.output_proj(attn_output)
        


# class TransformerBlock(nn.Module):
#     def __init__(
#         self,
#         d_model: int,
#         num_heads: int,
#         d_ff: int,
#         max_seq_len: int,
#         theta: float,
#         device=None,
#         dtype=None,
#     ):
#         super().__init__()
#         self.d_model = d_model
#         self.max_seq_len = max_seq_len

#         self.ln1 = Rmsnorm(d_model, device=device, dtype=dtype)
#         self.ln2 = Rmsnorm(d_model, device=device, dtype=dtype)
#         self.ffn = SwiGLU(d_model, d_ff, device=device, dtype=dtype)  
#         self.attn = CausalMultiHeadSelfAttention(
#             d_model=d_model,
#             num_heads=num_heads,
#             max_seq_len=max_seq_len,
#             theta=theta,
#             device=device,     
#         )

#     def load_weights(self, weights: dict[str, torch.Tensor], prefix: str = "") -> None:
#         """一次性装载本层的六个参数。attn 的四个投影权重不含在模块里，走 forward 入参。"""
#         with torch.no_grad():
#             self.ln1.weight.copy_(weights[prefix + "ln1.weight"])
#             self.ln2.weight.copy_(weights[prefix + "ln2.weight"])
#             self.ffn.w1.weight.copy_(weights[prefix + "ffn.w1.weight"])
#             self.ffn.w3.weight.copy_(weights[prefix + "ffn.w3.weight"])
#             self.ffn.w2.weight.copy_(weights[prefix + "ffn.w2.weight"])

#     def forward(
#         self,
#         x: torch.Tensor,
#         weights: dict[str, torch.Tensor],
#         prefix: str = "",
#     ) -> torch.Tensor:
#         seq_len = x.shape[-2]

#         q_w = weights[prefix + "attn.q_proj.weight"]
#         k_w = weights[prefix + "attn.k_proj.weight"]
#         v_w = weights[prefix + "attn.v_proj.weight"]
#         o_w = weights[prefix + "attn.output_proj.weight"]

#         token_positions = torch.arange(seq_len, device=x.device)   

#         h = x + self.attn(self.ln1(x), q_w, k_w, v_w, o_w, token_positions)
#         out = h + self.ffn(self.ln2(h))
#         return out

class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, max_seq_len, theta, device=None, dtype=None):
        super().__init__()
        self.ln1 = Rmsnorm(d_model, device=device, dtype=dtype)
        self.ln2 = Rmsnorm(d_model, device=device, dtype=dtype)
        self.ffn = SwiGLU(d_model, d_ff, device=device, dtype=dtype)
        self.attn = CausalMultiHeadSelfAttention(d_model, num_heads, max_seq_len, theta, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-norm 结构：先 Norm，再 Attention/FFN，再残差连接
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x

# class TransformerLM(nn.Module):
#     def __init__(
#         self,
#         vocab_size: int,
#         context_length: int,
#         d_model: int,
#         num_layers: int,
#         num_heads: int,
#         d_ff: int,
#         rope_theta: float,
#         device=None,
#         dtype=None,
#     ):
#         super().__init__()
#         self.vocab_size = vocab_size
#         self.context_length = context_length
#         self.d_model = d_model
#         self.num_layers = num_layers
#         self.token_embedding = Embedding(vocab_size,d_model,device=device,dtype=dtype)
#         self.layers = nn.ModuleList([
#             TransformerBlock(
#                 d_model, num_heads, d_ff, context_length, rope_theta,
#                 device=device, dtype=dtype,
#             )
#             for _ in range(num_layers)
#         ])
#         self.ln_final = Rmsnorm(d_model,device=device,dtype=dtype)
#         self.lm_head = Linear(d_model,vocab_size,device=device,dtype=dtype)
#     def load_weights(self, weights: dict[str, torch.Tensor], prefix: str = "") -> None:
#         with torch.no_grad():
#             self.token_embeddings.weight.copy_(weights[prefix + "token_embeddings.weight"])
#             self.ln_final.weight.copy_(weights[prefix + "ln_final.weight"])
#             self.lm_head.weight.copy_(weights[prefix + "lm_head.weight"])
#         for i,block in enumate(self.layers):
#             block.load_weights(weights,prefix=prefix + f"layers.{i}.")
#     def forward(
#         self,
#         in_indices: torch.Tensor,
#         weights: dict[str, torch.Tensor],
#         prefix: str = "",
#     ) -> torch.Tensor:
#         assert in_indices.shape[-1] <= self.context_length
#         x = self.token_embedding(in_indices)
#         for i,block in enumerate(self.layers):
#             x = block(x,weights,prefix=prefix + f"layers.{i}.")
#         x = self.ln_final(x)
#         out=self.lm_head(x)
        
#         return out


class TransformerLM(nn.Module):
    def __init__(self, vocab_size, context_length, d_model, num_layers, num_heads, d_ff, rope_theta, device=None, dtype=None):
        super().__init__()
        self.context_length = context_length
        self.token_embedding = Embedding(vocab_size, d_model, device=device, dtype=dtype)
        
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, context_length, rope_theta, device=device, dtype=dtype)
            for _ in range(num_layers)
        ])
        
        self.ln_final = Rmsnorm(d_model, device=device, dtype=dtype)
        self.lm_head = Linear(d_model, vocab_size, device=device, dtype=dtype)

    def forward(self, in_indices: torch.Tensor) -> torch.Tensor:
        assert in_indices.shape[-1] <= self.context_length
        
        # 查表
        x = self.token_embedding(in_indices)
        
        # 通过每一个 Transformer Block
        for layer in self.layers:
            x = layer(x)
            
        x = self.ln_final(x)
        return self.lm_head(x)

class AdamW(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2):
        # 检查参数合法性
        if lr < 0.0: raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0: raise ValueError(f"Invalid beta1: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0: raise ValueError(f"Invalid beta2: {betas[1]}")
        if not 0.0 <= eps: raise ValueError(f"Invalid eps: {eps}")
        if not 0.0 <= weight_decay: raise ValueError(f"Invalid weight_decay: {weight_decay}")
        
        # 将超参数打包成一个字典传给父类
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
                
        for group in self.param_groups:
            beta1, beta2 = group['betas']
            eps = group['eps']
            lr = group['lr']
            weight_decay = group['weight_decay']
            
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad
                
                # 1. 初始化状态（第一、第二动量，以及全局步数 t）
                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                
                # 2. t 加 1
                state['step'] += 1
                t = state['step']
                
                # 3. 计算动态学习率 alpha_t
                # 提示：alpha_t = lr * sqrt(1 - beta2**t) / (1 - beta1**t)
                # 【填空 A】：计算 bias_correction1 和 bias_correction2，然后算 alpha_t
                bias_correction1 = 1 - beta1 ** t
                bias_correction2 = 1 - beta2 ** t
                alpha_t = lr * (bias_correction2 ** 0.5) / bias_correction1
                
                # 4. 更新参数（第一次更新：Weight Decay）
                # 提示：p = p - lr * weight_decay * p
                # 【填空 B】
                p -= lr * weight_decay * p
                
                # 5. 更新一阶和二阶矩
                # 【填空 C】
                exp_avg = beta1 * exp_avg + (1 - beta1) * grad
                exp_avg_sq = beta2 * exp_avg_sq + (1 - beta2) * (grad ** 2)
                
                # 6. 更新参数（第二次更新：动量调整）
                # 提示：p = p - alpha_t * exp_avg / (sqrt(exp_avg_sq) + eps)
                # 【填空 D】
                p -= alpha_t * exp_avg / (exp_avg_sq.sqrt() + eps)
                
                # 别忘了把更新后的 exp_avg 和 exp_avg_sq 存回 state！
                state['exp_avg'] = exp_avg
                state['exp_avg_sq'] = exp_avg_sq

        return loss

