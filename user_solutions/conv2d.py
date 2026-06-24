import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ✏️ YOUR IMPLEMENTATION HERE

def my_conv2d(x, weight, bias=None, stride=1, padding=0): # no dilation (i.e., dilation=1)
    # extract patches, apply kernel, handle stride/padding
    # x: (B, C_in, H, W), weight: (C_out, C_in, kH, kW), bias: None or (C_out)
    # Returns: (B, C_out, H_out, W_out)
    B, C_in, H, W = x.shape
    C_out, _, kH, kW = weight.shape
    H_padded = H+2*padding
    W_padded = W+2*padding
    H_out = (H_padded - kH) // stride + 1
    W_out = (W_padded - kW) // stride + 1

    # F.unfold can handle pading & stride
    # extract patches using F.unfold(input, kernel_size, dilation=1, padding=0, stride=1)
    patches = F.unfold(x, kernel_size=(kH, kW), padding=padding, stride=stride)
        # patches.shape = (B, C_in * kH * kW, H_out * W_out)

    # apply kernel
    weight = weight.view(C_out, C_in * kH * kW)
        # or: weight = weight.view(C_out, -1), 说明dim=0是C_out、dim=1是-1即自动推断
    out = weight @ patches
        # out.shape = (B, C_out, H_out * W_out)
    out = out.view(B, C_out, H_out, W_out)
    # if bias: erro: 多元素 Tensor 不能直接转成布尔值
    if bias is not None:
        bias = bias.view(1, C_out, 1, 1) # shape = C_out -> 1*C_out*1*1
        out += bias
    return out
    

    