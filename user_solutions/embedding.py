import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ✏️ YOUR IMPLEMENTATION HERE

class MyEmbedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim):
        super().__init__()

        # trainable: use nn.Parameter instead of simple tensor/matrix
        # nn.Parameter需要传入tensor，而不仅仅是shape
        self.weight = nn.Parameter(torch.rand(num_embeddings, embedding_dim))

    def forward(self, indices):
        return self.weight[indices]