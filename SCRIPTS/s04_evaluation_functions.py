import numpy as np
import torch

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

def compute_run_acc(logits, labels):
    # returns how often we were correct/how often we were wrong
    # > "boolean units" =/=loss, although are usually linked
    _, pred = torch.max(logits.data, 1) #TODO: this is wrong
    return (pred == labels).sum().item()

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def autorec_loss(prediction, groundtruth, device, optimizer, regul=None):
    loss = torch.Tensor([0]).to(device)
    for i in range(len(groundtruth)):
        for j in range(len(groundtruth[i])):
            """99.0 is the value set for missing ratings"""
            if groundtruth[i][j] < 99.0:
                loss += (groundtruth[i][j] - prediction[i][j]) ** 2
    if regul:
        print(type(optimizer))
        V, W = optimizer
        loss += 0.5 * regul * (torch.linalg.norm(V, p='fro')+torch.linalg.norm(W, p='fro'))
    return loss
    # loss should  be of type, content : <class 'torch.Tensor'> tensor(0.9984, device='cuda:0', grad_fn=<MseLossBackward>)