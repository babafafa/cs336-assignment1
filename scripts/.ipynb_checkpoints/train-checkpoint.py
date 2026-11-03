import torch
import numpy as np
import os
import time
from cs336_basics.config import TrainConfig 
from cs336_basics.model import TransformerLM, AdamW 
from cs336_basics.data import get_batch
from cs336_basics.checkpoint import save_checkpoint,load_checkpoint
from cs336_basics.loss import cross_entropy
from cs336_basics.schedule import lr_cosine_schedule,gradient_clipping

def evaluate(model, data, vocab_size, batch_size, context_length, device, eval_steps=20):
    model.eval()
    total_loss = 0.0
    seed_rng = np.random.RandomState(42) 
    with torch.no_grad():
        for _ in range(eval_steps):
            start = seed_rng.randint(0, len(data) - context_length, size=batch_size)
            x_indices = start[:, None] + np.arange(context_length)
            y_indices = x_indices + 1
            
            X = torch.tensor(data[x_indices], dtype=torch.long, device=device)
            Y = torch.tensor(data[y_indices], dtype=torch.long, device=device)
            
            logits = model(X)
            loss = cross_entropy(logits.reshape(-1, vocab_size), Y.reshape(-1))
            total_loss += loss.item()

    model.train() 
    return total_loss / eval_steps
    

def train():

    cfg = TrainConfig()
    cfg.validate()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)
        
    model = TransformerLM(
        vocab_size=cfg.model.vocab_size, context_length=cfg.model.context_length, 
        d_model=cfg.model.d_model, num_layers=cfg.model.num_layers,
        num_heads=cfg.model.num_heads, d_ff=cfg.model.d_ff, rope_theta=cfg.model.rope_theta,
        device=device
    ).to(device)
    
    optimizer = AdamW(
        model.parameters(), lr=cfg.optimizer.lr,
        betas=(cfg.optimizer.beta1, cfg.optimizer.beta2), 
        eps=cfg.optimizer.eps, weight_decay=cfg.optimizer.weight_decay
    )

    if not os.path.exists(cfg.data.train_path):
        print(f"找不到数据文件 {cfg.data.train_path}，请先确认路径！")
        return
        
    val_data = None
    if os.path.exists(cfg.data.val_path):
        val_data = np.load(cfg.data.val_path, mmap_mode='r')
        print("成功加载验证集！")
    else:
        print("警告：未找到验证集，将跳过验证！")
        
    data = np.load(cfg.data.train_path, mmap_mode='r')

    start_step = 0
    if cfg.load_ckpt_path and os.path.exists(cfg.load_ckpt_path):
        start_step = load_checkpoint(cfg.load_ckpt_path, model, optimizer) + 1
        print(f"从第 {start_step} 步恢复训练！")

    model.train()
    start_time = time.time()
    
    for step in range(start_step, cfg.total_steps):
        current_lr = lr_cosine_schedule(
            step, cfg.optimizer.lr, cfg.optimizer.min_lr, 
            cfg.schedule.warmup_iters, cfg.schedule.cosine_cycle_iters
        )
        
        for param_group in optimizer.param_groups:
            param_group['lr'] = current_lr

        X, Y = get_batch(data, cfg.data.batch_size, cfg.model.context_length, device)
        
        logits = model(X)

        loss = cross_entropy(logits.reshape(-1, cfg.model.vocab_size), Y.reshape(-1))
        
        optimizer.zero_grad()
        loss.backward()
        
    
        gradient_clipping(model.parameters(), cfg.optimizer.grad_clip)
        optimizer.step()
        
        if step % cfg.log_interval == 0:
            elapsed = time.time() - start_time
            print(f"Step {step:05d} | Loss: {loss.item():.4f} | LR: {current_lr:.6f} | Time: {elapsed:.1f}s")
            
        if val_data is not None and step > 0 and step % cfg.eval_interval == 0:
            val_loss = evaluate(
                model, val_data, cfg.model.vocab_size, 
                cfg.data.batch_size, cfg.model.context_length, device, 
                cfg.eval_batches
            )
            print(f"--- [验证] Step {step:05d} | Val Loss: {val_loss:.4f} ---")
            
        if step > 0 and step % cfg.save_interval == 0:
            save_checkpoint(model, optimizer, step, cfg.save_ckpt_path)
            print(f"--> 模型已保存至 {cfg.save_ckpt_path}")

if __name__ == "__main__":
    train()