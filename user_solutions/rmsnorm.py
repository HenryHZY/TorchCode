import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ✏️ YOUR IMPLEMENTATION HERE

def rms_norm(x, weight, eps=1e-6):
    # x.shape = (B, D)
    # weight.shape = (D)
    rms = torch.sqrt(1/x.shape[-1]*torch.sum(x**2, dim=-1, keepdim=True) + eps) # (B, 1)
    # or: rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)

    # error: RMSNorm用的weight需要element-wise product
    # rms_norm = x/rms @ weight
    rms_norm = x/rms * weight
    return rms_norm