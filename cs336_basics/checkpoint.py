import torch
import os

def save_checkpoint(model, optimizer, iteration, out):
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "iteration": iteration,
    }
    # 1. 确保目录存在
    out_dir = os.path.dirname(out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
 
    torch.save(checkpoint, out)
   

def load_checkpoint(src, model, optimizer):
    checkpoint=torch.load(src,map_location="cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    return checkpoint["iteration"]