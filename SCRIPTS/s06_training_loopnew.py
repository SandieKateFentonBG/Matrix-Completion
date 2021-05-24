import numpy as np
from s05_running_lossnew import *

def training_loop(model, optimizer, mystudy, trainloader, testloader, VISU = True):

    best_val_acc = 0  #TODO : check this?np.inf?
    autorec_te_losses = np.zeros(mystudy.num_epochs)
    rmse_te_losses = np.zeros(mystudy.num_epochs)
    te_accuracies = np.zeros(mystudy.num_epochs)
    autorec_tr_losses = np.zeros(mystudy.num_epochs)
    autorec_tr_accuracies = np.zeros(mystudy.num_epochs)

    for epoch_nr in range(mystudy.num_epochs):

        # Train model
        tr_dict = run_training(trainloader, optimizer, model, mystudy)
        te_dict = run_testing(testloader, model, mystudy)

        te_loss = te_dict['autorec_te_loss']
        rmse_loss = te_dict['rmse_te_loss']
        te_acc = te_dict['te_acc']
        tr_loss = tr_dict['autorec_tr_loss']
        tr_acc = tr_dict['autorec_tr_acc']

        # Get validation results
        autorec_te_losses[epoch_nr] = te_loss
        rmse_te_losses[epoch_nr] = rmse_loss
        te_accuracies[epoch_nr] = te_acc
        autorec_tr_losses[epoch_nr] = tr_loss
        autorec_tr_accuracies[epoch_nr] = tr_acc

        if VISU:
            print("Epoch {}:".format(epoch_nr))
            print('>> TRAIN: Epoch {} completed | tr_loss: {:.4f} | tr_acc: {:.4f}'.format(epoch_nr, tr_loss, tr_acc))
            print('>> VALIDATION: Epoch {} | rmse_te_loss: {:.4f} | autorec_te_loss: {:.4f}'.format(epoch_nr, rmse_loss, te_loss))
            print('>> ACCURACY: Epoch {} | te_acc: {:.4f} | best_val_acc: {:.4f}'.format(epoch_nr, te_acc, best_val_acc))

        # Save model if higher accuracy
        if te_acc > best_val_acc:
            best_val_acc = te_acc #
            torch.save(model.state_dict(), './jester_model.pth') #TODO : what is this? './cifar_net.pth'?
            print('>> SAVE: Epoch {} | Model saved'.format(epoch_nr))

    if VISU:
        print('Training finished')

    loop_dict = dict()
    loop_dict['te_losses'] = autorec_te_losses
    loop_dict['rmse_te_losses'] = rmse_te_losses
    loop_dict['te_accuracies'] = te_accuracies
    loop_dict['tr_losses'] = autorec_tr_losses
    loop_dict['tr_accuracies'] = autorec_tr_accuracies
    loop_dict['best_val_acc'] = best_val_acc

    return loop_dict






