import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from s04_evaluation_functions import *
from s03_model_definition import *

def compute_tr_loss(dataloader, optimizer, model, device, regul = 0.5):

    # Train model
    running_loss = 0.0

    for batch_data in dataloader:  # the input image is also the label
        # Train model - start with one item - 7700*1 ratings

        batch_data = batch_data.to(device)

        # Predict and get loss
        pred = model(batch_data).to(device) #x/y - instantiation + forward prop > get first prediction +/- accurate -
        loss = autorec_loss(pred, batch_data, model, regul)  # x/y - get first delta tensor/error matrix

        # Update model
        optimizer.zero_grad()  # thetas a/b/c - fill weights with zeros at start; optimizer = torch.optim.Rprop
        loss.backward()  #x/hidden/y - computes the gradients in a chain rule #TODO: isn't loss a scalar? should I have another loss of dim 7700*1?
        optimizer.step() # thetas a/b/c updates all the weight from the b #TODO: is this only done for the indices that paticipated in the loss computation

        running_loss += loss.item()

    tr_loss = running_loss / len(dataloader.dataset)

    return tr_loss

    #1. optimizer.step() : performs a parameter update based on the current gradient
    # (stored in .grad attribute of a parameter) and the update rule.
    #2. Calling .backward() multiple times accumulates the gradient (by addition) for each parameter.
    # This is why you should call optimizer.zero_grad() after each .step() call.
    # Note that following the first .backward call, a second call is only possible
    # after you have performed another forward pass, unless you specify retain_graph=True


    """# Sparsen model
    V, W = optimizer # TODO : compute all weights / only update for observed inputs ?
    Vsparse = remove_missing_ratings_2D(V, batch_data, placeholder='99.')
    Wsparse = remove_missing_ratings_2D(W, batch_data, placeholder='99.') #TODO : check dim of W = 500*7700 or 7700*500?
    sparse_optimzer = Vsparse, Wsparse
    optimizer = sparse_optimzer # TODO : check that this is a "deepcopy" > sparsity is saved into optimizer weights
    #here drop-out structure could be interesting as a reference - works only with certain elements
    #https://pytorch.org/docs/stable/generated/torch.nn.Dropout.html"""


def compute_te_loss(testloader, model, device, regul):

    # Get validation results from testing set
    running_loss = 0
    with torch.no_grad():  # no .backward() needed
        for batch_data in testloader:
            batch_data = batch_data.to(device)
            output = model(batch_data)
            running_loss += autorec_loss(output, batch_data, model, regul)

    te_loss = running_loss / len(testloader.dataset)

    return te_loss

def compute_perc_acc(testloader, device, model, round = True):

    summed_perc = 0
    summed_ratings = 0
    with torch.no_grad():  # no .backward() needed
        for batch_data in testloader:
            batch_data = batch_data.to(device)
            bool_mask = batch_data != 99.0
            output = model(batch_data)

            if round:
                batch_data = torch.round(batch_data)
                output = torch.round(output)
            perc = torch.abs(batch_data - output) / (torch.ones_like(batch_data)*20)  #TODO :?
            masked_perc = perc * bool_mask
            rating_count = torch.ones_like(batch_data) * bool_mask
            summed_perc += torch.sum(masked_perc)
            summed_ratings += torch.sum(rating_count)


    print('total margin : ',summed_perc , 'for total ratings :', summed_ratings )
    return summed_perc / summed_ratings

def compute_perc_acc2(testloader, device, model, round = True):

    print(type(testloader.dataset), testloader.dataset.shape, testloader.dataset)
    mask = testloader.dataset != 99.0
    print(type(mask), mask.shape, mask)
    outputs = torch.zeros_like(mask)
    with torch.no_grad():
        for i in range(len(testloader.dataset)):
            batch_data = testloader.dataset[i].to(device)
            output = model(batch_data)
            outputs[i] = output
        preds = outputs * mask
        print(preds)
        truths = testloader.dataset * mask
        if round:
            preds = torch.round(preds)
            truths = torch.round(truths)
        diff = torch.sum(preds - truths/(preds + truths))
    return diff.item()

'''def xxcompute_perc_acc(testloader, device, model, n_digits = 0):

    running_perc = 0
    with torch.no_grad():  # no .backward() needed
        for batch_data in testloader:
            batch_data = batch_data.to(device)
            mask = batch_data != 99.0
            output = model(batch_data)
            preds = torch.round(output * mask * 10**n_digits) / (10**n_digits)
            rounded_truths = torch.round(batch_data * mask * 10**n_digits) / (10**n_digits)
            avs = torch.sum(preds - rounded_truths / (preds + rounded_truths))
            running_perc += avs
    return running_perc / len(testloader.dataset)'''

'''def compute_perc_acc2(testloader, device, model, n_digits=0):

    print(type(testloader.dataset), testloader.dataset.shape, testloader.dataset)
    mask = testloader.dataset != 99.0
    print(type(mask), mask.shape, mask)
    outputs = torch.zeros_like(mask)
    with torch.no_grad():
        for i in range(len(testloader.dataset)):
            batch_data = testloader.dataset[i].to(device)
            output = model(batch_data)
            outputs[i] = output
        dte = outputs * mask * 10**n_digits
        print(type(dte))
        preds = torch.round(dte) / (10**n_digits)
        truths = torch.round(testloader * mask * 10**n_digits) / (10**n_digits)
        diff = torch.sum(preds - truths/(preds + truths))
    return diff.item()'''

def xremove_missing_ratings_1D(full_tensor, batch_data, placeholder = '99.'):

    indices = [] # TODO : check
    for data in batch_data:
        indices = (batch_data == float(placeholder)).nonzero(as_tuple=True)[0]
    for i in indices :
        full_tensor[i] = 0
    return full_tensor

def xremove_missing_ratings_2D(full_tensor, batch_data, placeholder = '99.'):

    missing_indices = [] # TODO : check
    for data in batch_data:
        missing_indices = (batch_data == float(placeholder)).nonzero(as_tuple=True)[0]
    for i in missing_indices:
        full_tensor[i] = torch.zeros_like(full_tensor[i])
    return full_tensor


def running_loss_print(new_folder=False, folder = None):

    print('5. Running loss')
    if new_folder:
        from s09_helper_functions import mkdir_p
        mkdir_p(folder)
    if folder :
        with open(folder + "results.txt", 'a') as f:
            print('5. Running loss', file=f)
        f.close()







