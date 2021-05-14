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

        # Put data on device
        batch_data = batch_data.to(device)

        # Predict and get loss
        pred = model(batch_data).to(device) #x/y - instantiation + forward prop > get first prediction +/- accurate -
        loss = autorec_loss(pred, batch_data, device, optimizer, regul)  # x/y - get first delta tensor/error matrix

        # TODO:
        # pred = 7700*1 / model = nn with V (7700*500) W (500*7700)
        # pred = ? scalar loss corresponding to current weights for this item
        #  but optimizer only appears after..?

        # Update model
        optimizer.zero_grad()  # thetas a/b/c - fill weights with zeros at start; optimizer = torch.optim.Rprop
        loss.backward()  #x/hidden/y - computes the gradients in a chain rule #TODO: isn't loss a scalar? should I have another loss of dim 7700*1?

        # Sparsen model
        V, W = optimizer # TODO : compute all weights / only update for observed inputs ?
        Vsparse = remove_missing_ratings_2D(V, batch_data, placeholder='99.')
        Wsparse = remove_missing_ratings_2D(W, batch_data, placeholder='99.') #TODO : check dim of W = 500*7700 or 7700*500?
        sparse_optimzer = Vsparse, Wsparse
        optimizer = sparse_optimzer # TODO : check that this is a "deepcopy" > sparsity is saved into optimizer weights

        optimizer.step()  # thetas a/b/c updates all the weight from the bp #TODO : before or after the sparsening?

        running_loss += loss.item()

    tr_loss = running_loss / len(dataloader.dataset)

    return tr_loss

        #1. optimizer.step() : performs a parameter update based on the current gradient
        # (stored in .grad attribute of a parameter) and the update rule.
        #2. Calling .backward() multiple times accumulates the gradient (by addition) for each parameter.
        # This is why you should call optimizer.zero_grad() after each .step() call.
        # Note that following the first .backward call, a second call is only possible
        # after you have performed another forward pass, unless you specify retain_graph=True

def remove_missing_ratings_1D(full_tensor, batch_data, placeholder = '99.'):

    indices = [] # TODO : check
    for data in batch_data:
        indices = (batch_data == float(placeholder)).nonzero(as_tuple=True)[0]
    for i in indices :
        full_tensor[i] = 0
    return full_tensor

def remove_missing_ratings_2D(full_tensor, batch_data, placeholder = '99.'):

    missing_indices = [] # TODO : check
    for data in batch_data:
        missing_indices = (batch_data == float(placeholder)).nonzero(as_tuple=True)[0]
    for i in missing_indices:
        full_tensor[i] = torch.zeros_like(full_tensor[i])
    return full_tensor


def compute_te_loss(testloader, model, device, regul):

    # Get validation results from testing set
    running_loss = 0
    with torch.no_grad():  # no .backward() needed
        for batch_data in testloader:
            batch_data = batch_data.to(device)
            output = model(batch_data)
            running_loss += autorec_loss(output, batch_data, regul)

    te_loss = running_loss / len(testloader.dataset)

    return te_loss







