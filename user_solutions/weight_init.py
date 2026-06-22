import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ✏️ YOUR IMPLEMENTATION HERE

def kaiming_init(weight):
    # fill with normal(0, sqrt(2/fan_in))
    fan_in = weight.shape[1]
    std = math.sqrt(2/fan_in)
    with torch.no_grad():
        weight.normal_(0, std) # in-place
    return weight