import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ✏️ YOUR IMPLEMENTATION HERE

class MyDropout(nn.Module):
    def __init__(self, p=0.5):
        super().__init__()

        if not 0 <= p < 1:
            raise ValueError("p must be in [0, 1)")
            
        self.p = p

    def forward(self, x):
        if not self.training or self.p == 0: # or (self.p - 0)<1e-8
            return x
        else:
            # uniform distribution on the interval [0, 1)
            drop_matrix = torch.rand(x.shape) 
            # drop_matrix = torch.rand_like(x) # 等价于上面的

            mask = (drop_matrix>self.p).float()
            return x*mask*(1/(1-self.p))