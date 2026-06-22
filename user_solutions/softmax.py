import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ✏️ YOUR IMPLEMENTATION HERE

def my_softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    # x could be 1D, 2D, ...
    # if only 1D:
    #     exp_ = torch.exp(x - torch.max(x)) # cannot use max(x)
    #     sum_ = sum(exp_)
    #     out = exp_/sum_

    # exp(x) != exp(x-max(x)), but softmax(exp(x)) = softmax(exp(x-max(x)))

    # torch.max(x)基本与x.max()等价; torch.sum(x)基本与x.sum()等价; 
    # exp_ = torch.exp(x - x.max(dim=dim, keepdim=True).values)
    exp_ = torch.exp(x - torch.max(x, dim=dim, keepdim=True).values)
    # sum_ = exp_.sum(dim=dim, keepdim=True)
    sum_ = torch.sum(exp_, dim=dim, keepdim=True)
    out = exp_/sum_
    return out
