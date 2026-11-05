import torch
import numpy as np
import os
import time
import argparse
import wandb
from dataclasses import asdict
from cs336_basics.config import TrainConfig 
from cs336_basics.model import TransformerLM, AdamW 
from cs336_basics.data import get_batch
from cs336_basics.checkpoint import save_checkpoint,load_checkpoint
from cs336_basics.loss import cross_entropy
from cs336_basics.schedule import lr_cosine_schedule,gradient_clipping
from cs336_basics.logging_utils import make_run_dir, save_config, append_metrics

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
    parser=argparse.ArgumentParser()
    parser.add_argument("--run-name",type=str,default="default_run")
    parser.add_argument("--lr", type=float, default=None, help="Peak learning rate")
    parser.add_argument("--total-steps", type=int, default=None, help="Total training steps")
    parser.add_argument("--warmup-iters", type=int, default=None, help="Warmup iters")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size")
    parser.add_argument("--save-ckpt-path", type=str, default=None, help="Checkpoint path")

    args = parser.parse_args()

    # 3. 如果命令行传了参数，就覆盖掉默认的 cfg
    if args.run_name is not None:
        cfg.run_name = args.run_name
    if args.lr is not None:
        cfg.optimizer.lr = args.lr
        cfg.optimizer.min_lr = args.lr / 10.0

    if args.batch_size is not None:
        cfg.data.batch_size = args.batch_size
    if args.save_ckpt_path is not None:
        cfg.save_ckpt_path = args.save_ckpt_path
    if args.total_steps is not None:
        cfg.total_steps = args.total_steps
        cfg.schedule.cosine_cycle_iters = args.total_steps
    if args.warmup_iters is not None:
        cfg.schedule.warmup_iters = args.warmup_iters
    else:
        cfg.schedule.warmup_iters = int(cfg.total_steps * 0.05)

    # 4. 用 run_name 自动生成 checkpoint 路径，防止互相覆盖
    if args.save_ckpt_path is None:
        cfg.save_ckpt_path = f"ckpt/{cfg.run_name}.pt"
        
    cfg.validate()   
    device = "cuda" if torch.cuda.is_available() else "cpu"
    run_dir=make_run_dir(cfg.runs_root,cfg.run_name)
    save_config(cfg,run_dir)
        
    wandb.init(
    project="cs336-assignment1", 
    name=args.run_name, 
    config=asdict(cfg) # 自动把超参数记录下来
)

    
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
        raise FileNotFoundError(f"找不到训练数据文件：{cfg.data.train_path}。请检查路径！")
        
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
    start_time = time.perf_counter()
    try:
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
            grad_norm = gradient_clipping(model.parameters(), cfg.optimizer.grad_clip)
            optimizer.step()
            
            # 1. 整理 record 字典
            record = {
                "step": step, 
                "wall_time_s": time.perf_counter() - start_time,
                "tokens_seen": (step + 1) * cfg.data.batch_size * cfg.model.context_length,
                "train_loss": float(loss),
                "lr": float(current_lr),
                "grad_norm": float(grad_norm),
            }

            is_log_step = (step % cfg.log_interval == 0)
            is_eval_step = (val_data is not None and step > 0 and step % cfg.eval_interval == 0)

            if is_eval_step:
                val_loss = evaluate(
                    model, val_data, cfg.model.vocab_size, 
                    cfg.data.batch_size, cfg.model.context_length, device, 
                    cfg.eval_batches
                )
                record["val_loss"] = float(val_loss)

            # 2. 如果是记录节点，同步写入 Wandb 和 本地 jsonl
            if is_log_step or is_eval_step:
                if is_log_step:
                    print(f"Step {step:05d} | Loss: {loss.item():.4f} | LR: {current_lr:.6f} | Grad Norm: {grad_norm:.2f}")
                if is_eval_step:
                    print(f"--- [验证] Step {step:05d} | Val Loss: {record['val_loss']:.4f} ---")
                
                wandb_record = record.copy()
                wandb_record.pop("step", None) 
                wandb.log(wandb_record, step=step)
                
                append_metrics(run_dir, record)
                
            # 3. 存档
            if step > 0 and step % save_interval == 0:
                save_checkpoint(model, optimizer, step, cfg.save_ckpt_path)
                print(f"--> 模型已保存至 {cfg.save_ckpt_path}")

    except KeyboardInterrupt:
        print("\n检测到 Ctrl+C，正在安全停止训练...")
        # 可以在中断时也保存一个 checkpoint 保底
        save_checkpoint(model, optimizer, step, f"{cfg.save_ckpt_path}.interrupted")
        print(f"--> 中断模型已保存至 {cfg.save_ckpt_path}.interrupted")
        
    finally:
        # 无论怎么退出，都干净地关闭 wandb
        wandb.finish()
if __name__ == "__main__":
    torch.set_float32_matmul_precision('high')
    train()