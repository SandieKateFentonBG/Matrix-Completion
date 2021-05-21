
from s09_I_autorec_model import *


def identify_tuning_parameter():
    pass

def run_model_architectures(project, database, date, num_epochs, learning_rate,
                                   variable_dict, VISU = False, new_folder=False, folder = None, reference = None):
    regul_list = variable_dict['regularization_term']
    hidden_list = variable_dict['hidden_dim']

    studied_attr = []
    if len(regul_list) > 1:
        studied_attr = ['regularization_term']#TODO : change this i the referencing!
    if len(hidden_list) > 1:
        studied_attr.append('regularization_term')#TODO : change this i the referencing!

    AE_list = []
    for r in regul_list:
        for h in hidden_list:
            myAE = I_Autorec_model(project, database, date, selected_group, hidden_dim, num_epochs, learning_rate,
                                   regularization_term,
                                   studied_attr, VISU=VISU, new_folder=new_folder, folder=folder, reference=reference)
            myAE.
            AE_list.append(myAE)

    return AE_list

def return_model_results(AE_list):




def tune_model_architecture_hidden(project, database, date, selected_group, hidden_dim_list, num_epochs, learning_rate, regularization_term,
                                   studied_attr, VISU = False, new_folder=False, folder = None, reference = None):


    AE_list = []

    for hidden_dim in hidden_dim_list:
        myAE = I_Autorec_model(project, database, date, selected_group, hidden_dim, num_epochs, learning_rate, regularization_term,
                    studied_attr, VISU = VISU, new_folder=new_folder, folder = folder, reference = reference)
        AE_list.append(myAE)

        """
        [mystudy, x_train, x_test, x_val, trainloader, testloader, valloader, model, tr_losses, te_losses, rmse_loss, 
        best-val_losses, perc_acc] = model
        """

    autorec_loss_list = []
    perc_acc_list = []
    rmse_loss_list = []
    best_val_loss_list = []
    for AE in AE_list:
        autorec_loss_list.append(AE[9])
        rmse_loss_list.append(AE[10])
        best_val_loss_list.append(AE[11])
        perc_acc_list.append(AE[12])

    return AE_list, hidden_dim_list, autorec_loss_list, perc_acc_list, rmse_loss_list, best_val_loss_list

def tune_model_architecture_groups(project, database, date, selected_group_list, hidden_dim, num_epochs,
                                        learning_rate, regularization_term,
                                        studied_attr, VISU=False, new_folder=False, folder=None, reference=None):

    #TODO : how can this process be done with validationset and not train set? should be trained on training, and tested on validation !!

    AE_list = []

    if str(elem) == 'selected_group'# todo : def choose iterable()

    for selected_group in selected_group_list:
        myAE = I_Autorec_model(project, database, date, selected_group, hidden_dim, num_epochs, learning_rate,
                               regularization_term,
                               studied_attr, VISU=VISU, new_folder=new_folder, folder=folder, reference=reference )
        AE_list.append(myAE)

        """
        [mystudy, x_train, x_test, x_val, trainloader, testloader, valloader, model, tr_losses, te_losses, rmse_losses, best-val_losses,
         perc_acc] = model
        """

    autorec_loss_list = []
    perc_acc_list = []
    rmse_loss_list = []
    best_val_loss_list = []
    for AE in AE_list:
        autorec_loss_list.append(AE[9])
        rmse_loss_list.append(AE[10])
        best_val_loss_list.append(AE[11])
        perc_acc_list.append(AE[12])

    return AE_list, selected_group_list, autorec_loss_list, perc_acc_list, rmse_loss_list, best_val_loss_list

def hypertune_model_architecture_regul(project, database, date, selected_group, hidden_dim, num_epochs,
                                        learning_rate, regularization_term_list,
                                        studied_attr, VISU=False, new_folder=False, folder=None, reference=None):
    #TODO : how can this process be done with validationset and not train set? should be trained on training, and tested on validation !!

    AE_list = []

    for regularization_term in regularization_term_list:
        myAE = I_Autorec_model(project, database, date, selected_group, hidden_dim, num_epochs, learning_rate,
                               regularization_term,
                               studied_attr, VISU=VISU, new_folder=new_folder, folder=folder, reference=reference)
        AE_list.append(myAE)

        """
        [mystudy, x_train, x_test, x_val, trainloader, testloader, valloader, model, tr_losses, te_losses, rmse_losses, best-val_losses,
         perc_acc] = model
        """

    autorec_loss_list = []
    perc_acc_list = []
    rmse_loss_list = []
    best_val_loss_list = []
    for AE in AE_list:
        autorec_loss_list.append(AE[9])
        rmse_loss_list.append(AE[10])
        best_val_loss_list.append(AE[11])
        perc_acc_list.append(AE[12])

    return AE_list, regularization_term_list, autorec_loss_list, perc_acc_list, rmse_loss_list, best_val_loss_list





