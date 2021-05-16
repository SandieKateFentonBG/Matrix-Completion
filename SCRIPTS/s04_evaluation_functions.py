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


def autorec_loss(prediction, groundtruth, model, regul=None):
    loss = torch.Tensor([0]).to(groundtruth.device)
    # TODO consider replacing nested for loops by a mask, something like:
    # mask = groudtruth != 99
    # loss = torch.sum((groundtruth - prediction)**2 * mask)
    for i in range(len(groundtruth)):
        for j in range(len(groundtruth[i])):
            """99.0 is the value set for missing ratings"""
            if groundtruth[i][j] < 99.0:
                loss += (groundtruth[i][j] - prediction[i][j]) ** 2
    if regul:
        V, W = model.fci.weights, model.fco.weights #!! weights exclude biases!
        # get_weights - returns list and biasa
        # here you could add Vbias, Wbias through a reshape = model.fci.bias, model.fco.bias

        loss += 0.5 * regul * (torch.linalg.norm(V, p='fro')+torch.linalg.norm(W, p='fro'))
    return loss
    # loss should  be of type, content : <class 'torch.Tensor'> tensor(0.9984, device='cuda:0', grad_fn=<MseLossBackward>)