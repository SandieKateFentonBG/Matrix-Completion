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
    for batch_data, _ in dataloader:  # the input image is also the label
        # Train model - start with one item - 7700*1 ratings

        # Put data on device
        batch_data = batch_data.to(device)

        # Predict and get loss
        output = model(batch_data) #x/y - instantiation + forward prop > get first prediction +/- accurate -
        # output = 7700*1 / model = nn with V (7700*500) W (500*7700)

        loss = autorec_loss(output, batch_data, optimizer, regul) #x/y - get first delta tensor/error matrix
        # output = scalar loss corresponding to current weights for this item
        # TODO: but optimizer only appears after..?

        # Update model
        optimizer.zero_grad()  # thetas a/b/c - fill weights with zeros at start; optimizer = torch.optim.Rprop

        loss.backward()  #x/hidden/y - computes the gradients in a chain rule #TODO: but isn't this a scalar?
        # TODO : only update the weights with observed inputs / although we compute them for everything ?

        V, W = optimizer
        Vsparse = remove_missing_ratings_2D(V, batch_data, placeholder='99.')
        Wsparse = remove_missing_ratings_2D(W, batch_data, placeholder='99.') #TODO : check dim of W = 500*7700 or 7700*500?
        sparse_optimzer = Vsparse, Wsparse

        optimizer = sparse_optimzer # TODO : check that this is a "deepcopy" > sparsity is saved into optimizer weights
        optimizer.step()  # thetas a/b/c updates all the weight from the bp

        running_loss += loss.item()


    tr_loss = running_loss / len(dataloader.dataset)

    return tr_loss

    #Returns the value of this tensor as a standard Python number.
    #This only works for tensors with one element.

    #1. optimizer.step() : performs a parameter update based on the current gradient
    # (stored in .grad attribute of a parameter) and the update rule.
    #2. Calling .backward() multiple times accumulates the gradient (by addition) for each parameter.
    # This is why you should call optimizer.zero_grad() after each .step() call.
    # Note that following the first .backward call, a second call is only possible
    # after you have performed another forward pass, unless you specify retain_graph=True

def remove_missing_ratings_1D(full_tensor, batch_data, placeholder = '99.'):
    index = None
    for data in batch_data:
        indices = (batch_data == float(placeholder)).nonzero(as_tuple=True)[0]
    for i in indices :
        full_tensor[i] = 0
    return full_tensor

def remove_missing_ratings_2D(full_tensor, batch_data, placeholder = '99.'):
    index = None
    for data in batch_data:
        missing_indices = (batch_data == float(placeholder)).nonzero(as_tuple=True)[0]
    for i in missing_indices:
        full_tensor[i] = torch.zeros_like(full_tensor[i])
    return full_tensor


def compute_te_loss(testloader, model, device, optimizer, regul):
    # Get validation results from testing set
    running_loss = 0
    with torch.no_grad():  # TODO : what is this
        for batch_data, _ in testloader:
            batch_data = batch_data.to(device)
            output = model(batch_data)
            running_loss += autorec_loss(output, batch_data, optimizer, regul)

    te_loss = running_loss / len(testloader.dataset)

    return te_loss

def training_loop(model,learning_rate, num_epochs, trainloader, testloader, device, regul):

    # Instantiate components (weights, loss)                                            > start with zeros?
    loss_function = autorec_loss() #TODO: is this instantiation needed? loss vs objective?
    weights = torch.optim.Rprop(model.parameters(), learning_rate) #to compute weight matrix
    """optimizer/weights
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-2)
        etas (Tuple[float, float], optional): pair of (etaminus, etaplis), that #TODO:                  What is this etas?
            are multiplicative increase and decrease factors
            (default: (0.5, 1.2))
        step_sizes (Tuple[float, float], optional): a pair of minimal and
            maximal allowed step sizes (default: (1e-6, 50))
    #'Rprop does not support sparse gradients'
    #https://pytorch.org/docs/master/_modules/torch/optim/rprop.html#Rprop
    """
    best_val_loss = np.inf #TODO : check?

    tr_losses = np.zeros(num_epochs)
    te_losses = np.zeros(num_epochs)
    val_losses = np.zeros(num_epochs)

    for epoch_nr in range(num_epochs):

        print("Epoch {}:".format(epoch_nr))

        tr_loss = compute_tr_loss(trainloader, weights, model, device, regul)
        print('>> TRAIN: Epoch {} completed | tr_loss: {:.4f}'.format(
            epoch_nr, running_loss / len(trainloader.dataset)))

        te_loss = compute_tr_loss(testloader, weights, model, device, regul)
        print('>> VALIDATION: Epoch {} | te_loss: {:.4f}'.format(epoch_nr, te_loss))

        tr_losses[epoch_nr] = tr_loss
        te_losses[epoch_nr] = te_loss

    print('Training finished')
    return tr_losses, te_losses





