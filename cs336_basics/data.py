import torch
import math
import numpy as np

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

