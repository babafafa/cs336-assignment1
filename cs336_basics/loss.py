import torch
import math

def cross_entropy(
    logits: torch.Tensor,    
    targets: torch.Tensor,    
) -> torch.Tensor:
    # 1. 减最大值：数值稳定，且不改变数学结果
    max_val = logits.max(dim=-1, keepdim=True).values
    shifted = logits - max_val

    # 2. log(exp) 相互抵消：
    #    log p[t] = o[t] - M - log(sum_a exp(o[a] - M))
    lse = torch.log(torch.sum(torch.exp(shifted), dim=-1))          # (...)

    # 3. 取出正确词对应的那一个 logit
    target_logits = shifted.gather(-1, targets.unsqueeze(-1)).squeeze(-1)

    # 4. 负对数似然，对所有 batch 维求平均
    return -(target_logits - lse).mean()
