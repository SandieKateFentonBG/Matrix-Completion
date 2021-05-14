import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from s05_running_loss import *

def training_loop(model, weights, num_epochs, trainloader, testloader, device, regul):

    best_val_loss = np.inf

    tr_losses = np.zeros(num_epochs)
    te_losses = np.zeros(num_epochs)

    for epoch_nr in range(num_epochs):

        print("Epoch {}:".format(epoch_nr))

        # Train model
        tr_loss = compute_tr_loss(trainloader, weights, model, device, regul)
        print('>> TRAIN: Epoch {} completed | tr_loss: {:.4f}'.format(epoch_nr, tr_loss))

        # Get validation results
        te_loss = compute_tr_loss(testloader, weights, model, device, regul)
        print('>> VALIDATION: Epoch {} | te_loss: {:.4f}'.format(epoch_nr, te_loss))

        tr_losses[epoch_nr] = tr_loss
        te_losses[epoch_nr] = te_loss

        # Save model if best accuracy on validation dataset until now
        if te_loss > best_val_loss:
            best_val_loss = te_loss
            torch.save(model.state_dict(), './cifar_net.pth')
            print('>> SAVE: Epoch {} | Model saved'.format(epoch_nr))

    print('Training finished')
    return tr_losses, te_losses
