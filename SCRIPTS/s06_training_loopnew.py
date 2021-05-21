import numpy as np
from s05_running_loss import *

def training_loop(model, optimizer, num_epochs, number_groups = 1, trainloader, testloader, device, regul, folder = None, VISU = True,
                  reference = None):
    # TODO : check < number_groups = 1
    best_val_acc = np.inf #TODO : check this??
    autorec_te_losses = np.zeros(num_epochs) #TODO : or num_epochs * number_groups?
    rmse_te_losses = np.zeros(num_epochs)
    te_accuracies = np.zeros(num_epochs)
    autorec_tr_losses = np.zeros(num_epochs)
    autorec_tr_accuracies = np.zeros(num_epochs)

    for epoch_nr in range(num_epochs):
        # TODO : this could be wher you iterate over groups
        # Train model
        tr_dict = run_training(trainloader, optimizer, model, device, regul)
        te_dict = run_testing(testloader, model, device, regul)

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


        # Save model if best accuracy on validation dataset until now
        if te_acc > best_val_acc:
            best_val_acc = te_acc #TODO : check this??
            torch.save(model.state_dict(), './cifar_net.pth')

        if folder:
            with open(folder + reference + ".txt", 'a') as f:
                print("Epoch {}:".format(epoch_nr), file=f)
                print('>> TRAIN: Epoch {} completed | tr_loss: {:.4f}'.format(epoch_nr, tr_loss), file=f)
                print('>> VALIDATION: Epoch {} | rmse_te_loss: {:.4f}'.format(epoch_nr, rmse_loss), file=f)
                if te_loss > best_val_loss:
                    print('>> SAVE: Epoch {} | Model saved'.format(epoch_nr), file=f)
            f.close()

        if VISU:
            print("Epoch {}:".format(epoch_nr))
            print('>> TRAIN: Epoch {} completed | tr_loss: {:.4f}'.format(epoch_nr, tr_loss))
            print('>> VALIDATION: Epoch {} | rmse_te_loss: {:.4f}'.format(epoch_nr, rmse_loss))
            if te_loss > best_val_loss:
                print('>> SAVE: Epoch {} | Model saved'.format(epoch_nr))
    if VISU:
        print('Training finished')

    loop_dict = dict()
    loop_dict['autorec_te_losses'] = autorec_te_losses
    loop_dict['rmse_te_losses'] = rmse_te_losses
    loop_dict['te_accuracies'] = te_accuracies
    loop_dict['autorec_tr_losses'] = autorec_tr_losses
    loop_dict['autorec_tr_accuracies'] = autorec_tr_accuracies
    loop_dict['best_val_acc'] = best_val_acc

    return loop_dict


