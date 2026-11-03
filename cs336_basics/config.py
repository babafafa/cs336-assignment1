from dataclasses import dataclass, field
import os
import torch
from typing import Optional

@dataclass
class ModelConfig:
    vocab_size: int = 10000
    context_length: int = 256
    d_model: int = 512
    num_layers: int = 4
    num_heads: int = 16
    d_ff: int = 1344
    rope_theta: float = 10000.0

@dataclass
class OptimizerConfig:
    lr: float = 1e-3          
    min_lr: float = 1e-4
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8
    weight_decay: float = 0.1
    grad_clip: float = 1.0

@dataclass
class ScheduleConfig:
    warmup_iters: int = 1000
    cosine_cycle_iters: int = 40000

@dataclass
class DataConfig:
    train_path: str = "/root/autodl-tmp/assignment1-basics-main/data/processed/ts_train_ids.npy"
    val_path: str = "/root/autodl-tmp/assignment1-basics-main/data/processed/ts_valid_ids.npy"
    batch_size: int = 32

@dataclass
class TrainConfig:
    total_steps: int = 40000
    seed: int = 42
    load_ckpt_path: Optional[str] = None 
    save_ckpt_path: str = "ckpt/run1.pt"
    
    log_interval: int = 50
    eval_interval: int = 500
    save_interval: int = 1000
    eval_batches: int = 20
    
    model: ModelConfig = field(default_factory=ModelConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    schedule: ScheduleConfig = field(default_factory=ScheduleConfig)
    data: DataConfig = field(default_factory=DataConfig)

    def validate(self):
        assert self.model.d_model % self.model.num_heads == 0, \
            f"d_model ({self.model.d_model}) 必须被 num_heads ({self.model.num_heads}) 整除！"
        assert self.schedule.warmup_iters < self.schedule.cosine_cycle_iters, \
            f"预热步数 ({self.schedule.warmup_iters}) 必须小于余弦周期 ({self.schedule.cosine_cycle_iters})！"
        assert self.optimizer.min_lr <= self.optimizer.lr, "最小学习率不能大于最大学习率！"
        assert self.total_steps >= self.schedule.cosine_cycle_iters, "总步数必须大于等于余弦周期！"
        
        # 确保存档目录存在
        out_dir = os.path.dirname(self.save_ckpt_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)