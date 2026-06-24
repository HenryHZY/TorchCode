import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ✏️ YOUR IMPLEMENTATION HERE

def my_layer_norm(x, gamma, beta, eps=1e-5):
    # x.shape = (B, D)
    # gamma.shape = beta.shape = (B, D)

    mean = torch.mean(x, dim=-1, keepdim=True) # (B, 1)
    
    # Layernorm的std or var需要用unbiased=False
    # std  = torch.std(x, dim=-1, keepdim=True) # (B, 1)
    # std  = torch.std(x, dim=-1, keepdim=True, unbiased=False) # (B, 1)
    # out = gamma*(x-mean)/torch.sqrt(std**2 + eps) + beta
    # or
    var  = torch.var(x, dim=-1, keepdim=True, unbiased=False) # (B, 1)
    out = gamma*(x-mean)/torch.sqrt(var + eps) + beta

    return out