import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ✏️ YOUR IMPLEMENTATION HERE

class SwiGLUMLP(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        # Initialize gate_proj, up_proj, down_proj
        self.up_proj = nn.Linear(d_model, d_ff)
        self.gate_proj = nn.Linear(d_model, d_ff)
        self.down_proj = nn.Linear(d_ff, d_model)

    def my_silu(self, x):
        return x*torch.sigmoid(x)

    def forward(self, x):
        # down_proj(silu(gate_proj(x)) * up_proj(x))
        up = self.up_proj(x)
        
        gate = F.silu(self.gate_proj(x))
        # or:
        # gate = self.my_silu(self.gate_proj(x))
        
        down = self.down_proj(gate*up) # element-wise product
        return down