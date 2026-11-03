import torch
import math

def lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
) -> float:
    assert 0 <= warmup_iters < cosine_cycle_iters
    # 1. 预热阶段（线性增加）
    if it < warmup_iters:
        return it / warmup_iters * max_learning_rate
        
    # 2. 余弦退火阶段（平滑下降到最小值）
    elif it < cosine_cycle_iters:
        # 计算当前在退火阶段的进度（0 到 1 之间）
        progress = (it - warmup_iters) / (cosine_cycle_iters - warmup_iters)
        # 核心公式：min_lr + 0.5 * (1 + cos(pi * progress)) * (max_lr - min_lr)
        return min_learning_rate + 0.5 * (1 + math.cos(torch.pi * progress)) * (max_learning_rate - min_learning_rate)
        
    # 3. 退火结束阶段（保持最小值）
    else:
        return min_learning_rate

def gradient_clipping(parameters, max_l2_norm: float) -> float:
    params = [p for p in parameters if p.grad is not None]
    
    if len(params) == 0:
        return 0.0  # 如果没有梯度，返回 0

    total_norm_sq = params[0].grad.pow(2).sum()

    for p in params[1:]:
        total_norm_sq += p.grad.pow(2).sum()
        
    total_norm = torch.sqrt(total_norm_sq)

    eps = 1e-6
    clip_coef = torch.clamp(max_l2_norm / (total_norm + eps), max=1.0)

    for p in params:
        p.grad.mul_(clip_coef)
        

    return total_norm.item()