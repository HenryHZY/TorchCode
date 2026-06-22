import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ✏️ YOUR IMPLEMENTATION HERE

def clip_grad_norm(parameters, max_norm):
    # compute total norm, clip if needed, return original norm

    # error: not all p has gradient
    # sum_norm = torch.sqrt(sum(p.grad.norm()^2 for p in parameters)) 
    p_with_grad = [p for p in parameters if p.grad != None]
    if p_with_grad == []: return torch.tensor(0.0)

    # l2 norm = 平方根
    # p.grad.norm()是tensor
    total_norm = torch.sqrt(sum(p.grad.norm() ** 2 for p in p_with_grad))

    clip_coef = max_norm/total_norm
    
    # 把 clip_coef 的最大值限制为 1.0（当total_norm>max_norm)
    # clip_coef = torch.clamp(clip_coef, max=1.0)
    if clip_coef < 1:
        for p in parameters:
            if p.grad != None:
                p.grad*=clip_coef
                # p.grad.mul_(clip_coef) # 等价

    return total_norm
