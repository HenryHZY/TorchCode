import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ✏️ YOUR IMPLEMENTATION HERE

def my_batch_norm(
    x,
    gamma,
    beta,
    running_mean,
    running_var,
    eps=1e-5,
    momentum=0.1,
    training=True,
):
    # x.shape = (B, D)
    # gamma.shape = beta.shape = (B, D)
    if training:
        # error: 不能加keepdim=True，后面的updated in-place会报错
        mean = torch.mean(x, dim=0) # (D)
        var = torch.var(x, dim=0, unbiased=False) # (D)

        with torch.no_grad():
            # error: 没有updated in-place
            # running_mean = (1 - momentum) * running_mean + momentum * mean
            # running_var = (1 - momentum) * running_var + momentum * var
            running_mean.mul_(1 - momentum).add_(momentum * mean)
            running_var.mul_(1 - momentum).add_(momentum * var)
    else:
        mean = running_mean
        var = running_var

    norm = (x-mean)/torch.sqrt(var+eps)
    return gamma*norm+beta


