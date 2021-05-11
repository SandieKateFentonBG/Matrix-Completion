import numpy as np
import torch

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from evaluation_functions import *

def compute_tr_loss(dataloader):
    # Train model
    running_loss = 0.0
    for batch_data, _ in dataloader:  # the input image is also the label
        # Train model

        # Put data on device
        batch_data = batch_data.to(device)

        # Predict and get loss
        output = model(batch_data)
        loss = loss_function(output, batch_data)

        # Update model #TODO: this should only be done in places where we have ratings
        optimizer.zero_grad()  # fill weights with zeros at start

        loss.backward()  # computes the gradients in a chain rule #TODO: but isn't this a scalar?
        # TODO : only update the weights with observed inputs / although we compute them for everything ?
        optimizer.step()  # updates all the weight from the bp

        running_loss += loss.item()

    tr_loss = running_loss / len(dataloader.dataset)

    return tr_loss

def compute_te_loss(testloader):
    # Get validation results from testing set
    running_loss = 0
    with torch.no_grad():  # TODO : what is this
        for batch_data, _ in testloader:
            batch_data = batch_data.to(device)
            output = model(batch_data)
            running_loss += loss_function(output, batch_data).item()

    te_loss = running_loss / len(testloader.dataset)

    return te_loss

def training_loop(model,learning_rate, num_epochs, trainloader, testloader):

    #loss_function = autorec_loss() #TODO: loss vs objective?
    loss_function = nn.MSELoss() #TODO: remove

    # Instantiate components (optimizer, loss)
    optimizer = optim.Rprop(model.parameters(), learning_rate) #to compute weight matrix
    best_val_loss = np.inf #TODO : check?

    tr_losses = np.zeros(num_epochs)
    te_losses = np.zeros(num_epochs)
    val_losses = np.zeros(num_epochs)

    for epoch_nr in range(num_epochs):

        print("Epoch {}:".format(epoch_nr))

        tr_loss = compute_tr_loss(trainloader)
        print('>> TRAIN: Epoch {} completed | tr_loss: {:.4f}'.format(
            epoch_nr, running_loss / len(trainloader.dataset)))

        te_loss = compute_tr_loss(testloader)
        print('>> VALIDATION: Epoch {} | te_loss: {:.4f}'.format(epoch_nr, te_loss))

        tr_losses[epoch_nr] = tr_loss
        te_losses[epoch_nr] = te_loss

    print('Training finished')
    return tr_losses, te_losses