import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ✏️ YOUR IMPLEMENTATION HERE

def cross_entropy_loss(logits, targets):
    # log_probs = logits - logsumexp(...)
    
    # to extract x_y based on the index from targets
    batch_idx = torch.arange(logits.shape[0])
    x_y = logits[batch_idx, targets]
    # or based on torch.gather
    # x_y = torch.gather(logits, dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)

    # cal sum_exp_log using torch.logsumexp
    sum_exp_log = torch.logsumexp(logits, dim=-1, keepdim=True)

    logits = - (x_y - sum_exp_log)
    return logits.mean()