import numpy as np
import torch

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

def compute_run_acc(logits, labels):
    # returns how often we were correct/how often we were wrong
    # > "boolean units" =/=loss, although are usually linked
    _, pred = torch.max(logits.data, 1)
    return (pred == labels).sum().item()

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def autorec_loss(prediction, groundtruth, ): #TODO: write this

    """r_i = 'partially observed tensor'
    h(r_i, theta) = 'predicted rating'
    theta = [W, V] = 'weight matrix

    loss = (r-h)^2 + lamda*0.5*(W^2+V^2)
    '"""
    return sum(prediction - groundtruth) ^ 2 + 0.5 * regul * (theta[0]) ^ 2 + (theta[1]) ^ 2)
    pass
    torch.sum((encoder_i - decoder_i) ** 2)