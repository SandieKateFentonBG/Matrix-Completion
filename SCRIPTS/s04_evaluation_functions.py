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

    mask = groundtruth != 99.0
    full = torch.square(groundtruth - prediction)
    loss = torch.sum(full * mask).to(groundtruth.device)

    if regul:
        V_frob_norm = sum([(p ** 2).sum() for p in model.hidden.parameters()])
        W_frob_norm = sum([(q ** 2).sum() for q in model.predict.parameters()])

        loss += 0.5 * regul * (V_frob_norm + W_frob_norm)
    return loss
    # loss should  be of type, content : <class 'torch.Tensor'> tensor(0.9984, device='cuda:0', grad_fn=<MseLossBackward>)