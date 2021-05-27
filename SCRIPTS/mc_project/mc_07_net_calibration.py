import numpy as np

from mc_06_loss_functions import *

def net_calibration(model, optimizer, mystudy, trainloader, testloader, VISU = True):

    best_val_acc = 0  #TODO : check this?np.inf?
    autorec_te_losses = np.zeros(mystudy.num_epochs)
    rmse_te_losses = np.zeros(mystudy.num_epochs)
    te_accuracies = np.zeros(mystudy.num_epochs)
    autorec_tr_losses = np.zeros(mystudy.num_epochs)
    autorec_tr_accuracies = np.zeros(mystudy.num_epochs)

    for epoch_nr in range(mystudy.num_epochs):

        # Train model
        tr_dict = training_loss(trainloader, optimizer, model, mystudy)
        autorec_tr_losses[epoch_nr] = tr_dict['autorec_tr_loss']
        autorec_tr_accuracies[epoch_nr] = tr_dict['autorec_tr_acc']
        # Get validation results
        te_dict = testing_loss(testloader, model, mystudy)
        autorec_te_losses[epoch_nr] = te_dict['autorec_te_loss']
        rmse_te_losses[epoch_nr] = te_dict['rmse_te_loss']
        te_accuracies[epoch_nr] = te_dict['te_acc']

        if VISU:
            print("Epoch {}:".format(epoch_nr))
            print('>> TRAIN: Epoch {} completed | tr_loss: {:.4f} | tr_acc: {:.4f}'.format(epoch_nr, tr_loss, tr_acc))
            print('>> VALIDATION: Epoch {} | rmse_te_loss: {:.4f} | autorec_te_loss: {:.4f}'.format(epoch_nr, rmse_loss, te_loss))
            print('>> ACCURACY: Epoch {} | te_acc: {:.4f} | best_val_acc: {:.4f}'.format(epoch_nr, te_acc, best_val_acc))

        # Save model if higher accuracy
        if te_accuracies[epoch_nr] > best_val_acc:
            best_val_acc = te_accuracies[epoch_nr] #

            torch.save({  #TODO : check this only saves best result
                'epoch': epoch_nr,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'tr_losses': autorec_tr_losses[epoch_nr],
                'te_losses': autorec_te_losses[epoch_nr],
                'rmse_te_losses':rmse_te_losses[epoch_nr],
                'te_accuracies': te_accuracies[epoch_nr],
                'tr_accuracies': autorec_tr_accuracies[epoch_nr],
                'best_val_acc': best_val_acc,
            }, mystudy.output_path + mystudy.reference)

            print('>> SAVE: Epoch {} | Model saved'.format(epoch_nr))

    if VISU:
        print('Training finished')

    loss_dict = dict()
    loss_dict['te_losses'] = autorec_te_losses
    loss_dict['rmse_te_losses'] = rmse_te_losses
    loss_dict['te_accuracies'] = te_accuracies
    loss_dict['tr_losses'] = autorec_tr_losses
    loss_dict['tr_accuracies'] = autorec_tr_accuracies
    loss_dict['best_val_acc'] = best_val_acc

    return loss_dict



def net_calibration_print(loss_dict, output_path=None, reference = None, VISU=False):

    for k, v in loss_dict.items():
        if VISU:
            print(' ', k, ' : ', v)
        if output_path:
            with open(output_path + reference + ".txt", 'a') as f:
                print(' ', k, ' : ', v, file=f)
            f.close()


