import numpy as np
from mc_tr_05_Autorec_net import *
from mc_tr_07_loss_functions import *

def net_calibration(model, optimizer, mydata, myparams, reference, trainloader, testloader, VISU = False, threshold = None):

    best_val_acc = 0  #TODO : check this?np.inf?
    autorec_te_losses = np.zeros(myparams.num_epochs)
    rmse_te_losses = np.zeros(myparams.num_epochs)
    te_accuracies = np.zeros(myparams.num_epochs)
    autorec_tr_losses = np.zeros(myparams.num_epochs)
    autorec_tr_accuracies = np.zeros(myparams.num_epochs)

    for epoch_nr in range(myparams.num_epochs):

        # if epoch_nr==8:
        #     a = 2
        # Train model
        tr_dict = training_loss(trainloader, optimizer, model, myparams, threshold = threshold)
        autorec_tr_losses[epoch_nr] = tr_dict['autorec_tr_loss']
        autorec_tr_accuracies[epoch_nr] = tr_dict['autorec_tr_acc']
        # Get validation results
        te_dict = testing_loss(testloader, model, myparams, threshold = threshold)
        autorec_te_losses[epoch_nr] = te_dict['autorec_te_loss']
        rmse_te_losses[epoch_nr] = te_dict['rmse_te_loss']
        te_accuracies[epoch_nr] = te_dict['te_acc']

        if VISU:
            print("Epoch {}:".format(epoch_nr))
            print('>> TRAIN: Epoch {} completed | tr_loss: {:.4f} | tr_acc: {:.4f}'.format(epoch_nr, tr_dict['autorec_tr_loss'], tr_dict['autorec_tr_acc']))
            print('>> VALIDATION: Epoch {} | rmse_te_loss: {:.4f} | autorec_te_loss: {:.4f}'.format(epoch_nr, te_dict['rmse_te_loss'], te_dict['autorec_te_loss']))
            print('>> ACCURACY: Epoch {} | te_acc: {:.4f} | best_val_acc: {:.4f}'.format(epoch_nr, te_dict['te_acc'], best_val_acc))

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
            }, mydata.output_path + reference)

            best_score_dict = dict()
            best_score_dict['te_losses'] = autorec_te_losses[epoch_nr]
            best_score_dict['rmse_te_losses'] = rmse_te_losses[epoch_nr]
            best_score_dict['te_accuracies'] = te_accuracies[epoch_nr]
            best_score_dict['tr_losses'] = autorec_tr_losses[epoch_nr]
            best_score_dict['tr_accuracies'] = autorec_tr_accuracies[epoch_nr]
            best_score_dict['best_val_acc'] = best_val_acc

            print('>> SAVE: Epoch {} | Model saved'.format(epoch_nr))

    if VISU:
        print('Training finished')

    loss_evolution_dict = dict()
    loss_evolution_dict['te_losses'] = autorec_te_losses
    loss_evolution_dict['rmse_te_losses'] = rmse_te_losses
    loss_evolution_dict['te_accuracies'] = te_accuracies
    loss_evolution_dict['tr_losses'] = autorec_tr_losses
    loss_evolution_dict['tr_accuracies'] = autorec_tr_accuracies
    loss_evolution_dict['best_val_acc'] = best_val_acc

    return loss_evolution_dict, best_score_dict

def loss_calibration_print(loss_dict, reference, output_path=None, VISU=False):

    for k, v in loss_dict.items():
        if VISU:
            print(' ', k, ' : ', v)
        if output_path:
            with open(output_path + reference + ".txt", 'a') as f:
                print(' ', k, ' : ', v, file=f)
            f.close()



