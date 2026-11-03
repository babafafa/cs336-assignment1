import torch
import numpy as np
from cs336_basics.nn_utils import softmax


def top_p_filter(probs, top_p):
    if top_p >= 1.0:
        return probs
    
    # 1. 降序排序
    sorted_probs, sorted_indices = torch.sort(probs, dim=-1, descending=True)
    
    # 2. 计算累加和
    cumsum_probs = torch.cumsum(sorted_probs, dim=-1)
    
    # 3. 找到需要砍掉的词
    mask = (cumsum_probs - sorted_probs) > top_p
    
    # 4. 将不要的词置零
    sorted_probs[mask] = 0.0
    
    # 5. 重新归一化
    sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)
    
    # 6. 还原回原来的词汇顺序
    filtered_probs = torch.zeros_like(probs)
    filtered_probs.scatter_(dim=-1, index=sorted_indices, src=sorted_probs)
    
    return filtered_probs
    
@torch.no_grad()
def generate(model, input_ids, max_new_tokens, temperature=1.0, top_p=1.0, eos_token_id=None):
    """
    input_ids: 形状 (batch_size, seq_len) 的张量
    """
    model.eval()
    
    # 容错处理：防止温度设为 0 导致除零报错
    if temperature <= 0:
        temperature = 1e-5
    finished_mask = torch.zeros(batch_size, dtype=torch.bool, device=device)
    for _ in range(max_new_tokens):
        current_input_ids = input_ids[:, -model.context_length:] 
        logits=model(current_input_ids)
        next_token_logits=logits[:,-1,:]
        next_token_logits=next_token_logits/temperature
        probs=softmax(next_token_logits,dim=-1)
        probs=top_p_filter(probs,top_p)
    
        next_token = torch.multinomial(probs, num_samples=1)
            next_token = torch.where(finished_mask.unsqueeze(1), 
                             torch.tensor(eos_token_id, device=device), 
                             next_token)
                             
        # 更新结束标记
        finished_mask = finished_mask | (next_token.squeeze(1) == eos_token_id)
        
        # 拼接
        input_ids = torch.cat([input_ids, next_token], dim=-1)
        
        # 如果 batch 里所有句子都结束了，提前 break
        if finished_mask.all():
            break
        input_ids = torch.cat([input_ids, next_token], dim=-1)
        if eos_token_id is not None and (next_token==eos_token_id).all().item():
            break;
    return input_ids