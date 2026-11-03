from dataclasses import dataclass, field

# 1. 模型结构配置
@dataclass
class ModelConfig:
    vocab_size: int = 10000
    context_length: int = 256
    d_model: int = 512
    num_layers: int = 4
    num_heads: int = 16
    d_ff: int = 1344
    rope_theta: float = 10000.0

# 2. 优化器配置
@dataclass
class OptimizerConfig:
    max_lr: float = 1e-3
    min_lr: float = 1e-4
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8
    weight_decay: float = 0.1
    grad_clip: float = 1.0

# 3. 学习率调度配置
@dataclass
class ScheduleConfig:
    warmup_iters: int = 1000
    cosine_cycle_iters: int = 40000

# 4. 数据配置
@dataclass
class DataConfig:
    train_path: str = "tinystories_train_ids.npy"
    val_path: str = "tinystories_valid_ids.npy"
    batch_size: int = 32

# 5. 运行控制配置
@dataclass
class TrainConfig:
    total_steps: int = 40000
    seed: int = 42
    load_ckpt_path: str = None
    save_ckpt_path: str = "ckpt/run1.pt"
    
    # 把前面四个也打包进来
    model: ModelConfig = field(default_factory=ModelConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    schedule: ScheduleConfig = field(default_factory=ScheduleConfig)
    data: DataConfig = field(default_factory=DataConfig)

    # --- 极其重要的：统一校验 ---
    def validate(self):
        assert self.model.d_model % self.model.num_heads == 0, "d_model 必须被 num_heads 整除！"
        assert self.schedule.warmup_iters < self.schedule.cosine_cycle_iters, "预热步数必须小于总步数！"
        assert self.model.vocab_size >= 256, "词表大小不对！"