import numpy as np
from SCRIPTS.v2.s05_running_loss import *

def training_loop(model, optimizer, num_epochs, trainloader, testloader, device, regul, folder = None, VISU = True,
                  reference = None):

    best_val_loss = np.inf
    tr_losses = np.zeros(num_epochs)
    te_losses = np.zeros(num_epochs)
    
    for epoch_nr in range(num_epochs):
        # Train model
        tr_loss = autorec_tr_loss(trainloader, optimizer, model, device, regul)
        te_loss, rmse_loss = compute_te_loss(testloader, model, device, regul)
        # Get validation results
        tr_losses[epoch_nr] = tr_loss
        te_losses[epoch_nr] = te_loss
        # Save model if best accuracy on validation dataset until now
        if te_loss > best_val_loss:
            best_val_loss = te_loss
            torch.save(model.state_dict(), './cifar_net.pth')

        if folder:
            with open(folder + reference + ".txt", 'a') as f:
                print('6. Training Loop', file=f)
                print("Epoch {}:".format(epoch_nr), file=f)
                print('>> TRAIN: Epoch {} completed | tr_loss: {:.4f}'.format(epoch_nr, tr_loss), file=f)
                print('>> VALIDATION: Epoch {} | te_loss: {:.4f}'.format(epoch_nr, te_loss), file=f)
                if te_loss > best_val_loss:
                    print('>> SAVE: Epoch {} | Model saved'.format(epoch_nr), file=f)
            f.close()

        if VISU:
            print('6. Training Loop')
            print("Epoch {}:".format(epoch_nr))
            print('>> TRAIN: Epoch {} completed | tr_loss: {:.4f}'.format(epoch_nr, tr_loss))
            print('>> VALIDATION: Epoch {} | te_loss: {:.4f}'.format(epoch_nr, te_loss))
            if te_loss > best_val_loss:
                print('>> SAVE: Epoch {} | Model saved'.format(epoch_nr))
    if VISU:
        print('Training finished')
    return tr_losses, te_losses, best_val_loss #add best val loss?


