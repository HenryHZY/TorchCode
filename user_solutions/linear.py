import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ✏️ YOUR IMPLEMENTATION HERE

class SimpleLinear:
    def __init__(self, in_features: int, out_features: int):
        # Initialize weight and bias
        self.weight = nn.Parameter(torch.rand(out_features, in_features)*(1/math.sqrt(in_features)))
        self.bias = nn.Parameter(torch.zeros(out_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Compute y = x @ W^T + b
        # A @ B：矩阵乘法（matmul）
        # A * B：逐元素乘（element-wise / Hadamard 乘积）
        y = x@self.weight.T + self.bias
        return y