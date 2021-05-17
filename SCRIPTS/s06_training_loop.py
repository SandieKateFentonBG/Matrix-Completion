import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from s05_running_loss import *

def training_loop(model, optimizer, num_epochs, trainloader, testloader, device, regul, folder = None):

    print('6. Training Loop')
    best_val_loss = np.inf
    tr_losses = np.zeros(num_epochs)
    te_losses = np.zeros(num_epochs)

    for epoch_nr in range(num_epochs):

        with open(folder + "results.txt", 'a') as f:

            print("Epoch {}:".format(epoch_nr))
            print("Epoch {}:".format(epoch_nr), file=f)

            # Train model
            tr_loss = compute_tr_loss(trainloader, optimizer, model, device, regul)
            print('>> TRAIN: Epoch {} completed | tr_loss: {:.4f}'.format(epoch_nr, tr_loss))
            print('>> TRAIN: Epoch {} completed | tr_loss: {:.4f}'.format(epoch_nr, tr_loss), file=f)

            # Get validation results
            te_loss = compute_te_loss(testloader, model, device, regul)
            print('>> VALIDATION: Epoch {} | te_loss: {:.4f}'.format(epoch_nr, te_loss))
            print('>> VALIDATION: Epoch {} | te_loss: {:.4f}'.format(epoch_nr, te_loss), file=f)

            tr_losses[epoch_nr] = tr_loss
            te_losses[epoch_nr] = te_loss

            # Save model if best accuracy on validation dataset until now
            if te_loss > best_val_loss:
                best_val_loss = te_loss
                torch.save(model.state_dict(), './cifar_net.pth')
                print('>> SAVE: Epoch {} | Model saved'.format(epoch_nr))
                print('>> SAVE: Epoch {} | Model saved'.format(epoch_nr), file=f)

        f.close()

    print('Training finished')
    return tr_losses, te_losses


