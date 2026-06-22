import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ✏️ YOUR IMPLEMENTATION HERE

def my_gelu(x):
    out = x*0.5*(1+torch.erf(x/math.sqrt(2)))
    # out = x*0.5*(1+torch.erf(x/torch.sqrt(2))) # Note: 2 is not a tensor
    return out