import json
from cs336_basics.logging_utils import make_run_dir, save_config, append_metrics
from cs336_basics.config import TrainConfig

# 1. 用真实 TrainConfig
cfg = TrainConfig()
cfg.validate()          # 顺便验证 validate 不崩

# 2. 建 run 目录 + 存配置
d = make_run_dir(cfg.runs_root, "test")
save_config(cfg, d)

# 3. 追加一条 metrics
append_metrics(d, {"step": 0, "loss": 1.23})

# 4. 读回验证
loaded = json.load(open(d / "config.json"))
print("config.json:")
print(json.dumps(loaded, indent=2, ensure_ascii=False))

print("\nmetrics.jsonl:")
print(open(d / "metrics.jsonl").read())
