import torch
import math


def get_batch(x, batch_size, context_length, device):
    # 1. 随机生成 batch_size 个起点位置
    start = np.random.randint(0,len(x)-context_length,size=batch_size)    
    # 2. 根据起点切出 X 和 Y
    x_indices = start[:, None] + np.arange(context_length)
    y_indices = x_indices + 1

    X_np = x[x_indices]
    Y_np = x[y_indices]
    
    # 3. 转换为 PyTorch 张量，并放到指定 device 上
    X = torch.tensor(X_np,dtype=torch.long).to(device)
    Y = torch.tensor(Y_np,dtype=torch.long).to(device)
    
    return X, Y


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