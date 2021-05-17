import numpy as np
import torch

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

def compute_run_acc(logits, labels): #TODO: convert this for my own project
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

def evaluation_function_print(model, loss_function = 'custom autorec loss', new_folder=False, folder = None):

    print('4. Evaluation function')
    print("number of trainable parameters: {}".format(count_parameters(model)))
    print('loss function : ', loss_function)
    if new_folder:
        from s09_helper_functions import mkdir_p
        mkdir_p(folder)
    if folder :
        with open(folder + "results.txt", 'a') as f:
            print('4. Evaluation function', file=f)
            print("number of trainable parameters: {}".format(count_parameters(model)), file=f)
            print('loss function : ', loss_function, file=f)
        f.close()

