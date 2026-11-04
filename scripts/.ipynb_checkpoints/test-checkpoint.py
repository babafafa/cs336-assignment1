import torch
import numpy as np
from cs336_basics.config import TrainConfig
from cs336_basics.model import TransformerLM
from cs336_basics.data import get_batch
from cs336_basics.loss import cross_entropy

def main():
    cfg = TrainConfig()
    cfg.validate()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 1. 实例化模型（千万不要加载 checkpoint！）
    model = TransformerLM(
        vocab_size=cfg.model.vocab_size, context_length=cfg.model.context_length, 
        d_model=cfg.model.d_model, num_layers=cfg.model.num_layers,
        num_heads=cfg.model.num_heads, d_ff=cfg.model.d_ff, rope_theta=cfg.model.rope_theta,
        device=device
    ).to(device)
    
    # 2. 加载数据（只取一个 batch）
    data = np.load(cfg.data.train_path, mmap_mode='r')
    x, y = get_batch(data, cfg.data.batch_size, cfg.model.context_length, device)
    
    # 3. 前向传播（不需要优化器，不需要 backward）
    with torch.no_grad():
        logits = model(x)
        
        # 4. 算 Loss（记得 reshape！）
        loss = cross_entropy(logits.reshape(-1, cfg.model.vocab_size), y.reshape(-1))
        
    print(f"初始 Loss 是: {loss.item():.4f}")
    print(f"理论预期值: {np.log(cfg.model.vocab_size):.4f}")
    print("如果两者非常接近，说明模型初始化、数据加载和 Loss 函数全是对的！")

if __name__ == "__main__":
    main()