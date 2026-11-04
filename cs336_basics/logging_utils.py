import json
from dataclasses import asdict
import datatime
from pathlib import Path

def make_run_dir(root, name) -> Path:
    t=datatime.now().strftime("%Y%m%d-%H%M")
    run_dir=Path(root)/f"{t}_{name}"
    run_dir.mkdir(parents=True,exist_ok=False)
    return run_dir


def save_config(config, run_dir):



def append_metrics(run_dir, record):
