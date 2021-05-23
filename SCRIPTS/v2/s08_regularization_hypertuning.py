
from s09_I_autorec_model import *


def tune_model_architecture_hidden(project, database, date, selected_group, hidden_dim_list, num_epochs, learning_rate, regularization_term,
                                   studied_attr, VISU = False, new_folder=False, folder = None, reference = None):

    #TODO : how can this process be done with validationset and not train set? should be trained on training, and tested on validation !!

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

def plot_x_y_graph(x_list, y_list, x_label, y_label, title=None, folder=None, VISU=False):

    import matplotlib.pyplot as plt

    plt.figure()
    plt.plot(xdata = x_list, ydata = y_list,  label='Validation')
    if not title:
        title = x_label + 'as a function of' + y_label
    plt.title(reference)
    plt.ylabel(y_label)
    plt.xlabel(x_label)
    plt.legend()
    if folder:
        plt.savefig(folder + title + '.png')
    if VISU:
        plt.show()


def plot_sns_graph(x_list, y_list, x_label, y_label, title=None, figure_size=(12,15), folder=None, plot=False):
    import seaborn as sns
    import matplotlib.pyplot as plt
    import pandas as pd
    df = pd.DataFrame(list(zip(x_list, y_list)), columns =[x_label, y_label])
    fig, ax = plt.subplots(figsize=figure_size)
    if not title :
        title = x_label + 'as a function of' + y_label
    ax.set_title(title)
    sns.scatterplot(data=df, x=x_label, y=y_label, hue=y_label)
    if folder :
        plt.savefig(folder + '/'+ x_label + '-' + y_label +'.png')
    if plot:
        plt.show()


