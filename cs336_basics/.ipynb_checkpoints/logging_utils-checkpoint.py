import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

def make_run_dir(root:str, name:str) -> Path:
    t = datetime.now().strftime("%Y%m%d-%H%M%S")   # 加秒,减少冲突
    run_dir = Path(root) / f"{t}_{name}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir

def save_config(config, run_dir:str):
    cfg_dict=asdict(config)
    with open(run_dir/"config.json","w",encoding="utf-8") as f:
        json.dump(cfg_dict,f,indent=2,ensure_ascii=False)


def append_metrics(run_dir:str, record:dict) -> None:
    with open(run_dir/"metrics.jsonl","a",encoding="utf-8") as f:
        f.write(json.dumps(record,ensure_ascii=False)+"\n")
        f.flush()
