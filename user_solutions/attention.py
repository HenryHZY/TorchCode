import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ✏️ YOUR IMPLEMENTATION HERE

def scaled_dot_product_attention(Q, K, V):
    x = Q @ K.transpose(-2, -1)
        # or: x = Q @ K.transpose(-1, -2), 调换dim=-1和dim=-2
        # or: x = Q @ K.mT, mT可以调换最后两个dim，T会调换所有dim
    x = x/math.sqrt(K.shape[-1])
        # shape = (batch, seq_q, seq_k)
    attn = torch.softmax(x, dim=-1)
        # shape = (batch, seq_q, seq_k)
    out = attn @ V
    return out