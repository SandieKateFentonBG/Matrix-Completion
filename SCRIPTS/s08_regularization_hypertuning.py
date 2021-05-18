
from s10_I_autorec_model import *

# TODO : all this should be done for all 5 groups > then what?

def hypertune_model_parameters_regularization(model, reg_list):
    # TODO : regularization strength only influences the loss value > should not be hypertuned on a fix weight matrix but should be used to optimize weights
    [mystudy, x_train, x_test, x_val, trainloader, testloader, valloader, model, tr_losses, te_losses, val_losses,
     perc_acc] = model

    loss_list = []
    for val in reg_list:
        computed_loss = compute_te_loss(valloader, model, mystudy.device, val)
        loss_list.append(computed_loss)

    perc_acc = compute_perc_acc(valloader, mystudy.device, model) #TODO : REMOVE : should be the same for all

    return reg_list, loss_list, perc_acc #TODO : RMSE instead of loss; percentage acc instead of loss


def hypertune_model_architecture_hidden(project, database, date, selected_group, hidden_dim_list, num_epochs, learning_rate, regularization_term,
                    studied_attr, VISU = False, new_folder=False, folder = None, reference = None):

    AE_list = [] #TODO : how can this process be done with validationset and not train set?

    for hidden_dim in hidden_dim_list
        myAE = I_Autorec_model(project, database, date, selected_group, hidden_dim, num_epochs, learning_rate, regularization_term,
                    studied_attr, VISU = False, new_folder=False, folder = None, reference = None)
        AE_list.append(myAE)

    """
    [mystudy, x_train, x_test, x_val, trainloader, testloader, valloader, model, tr_losses, te_losses, val_losses,
     perc_acc] = model
    """"

    loss_list = []
    perc_acc_list = []
    for AE in AE_list:
        loss_list.append(AE[9])
        perc_acc_list.append(AE[11])

    return AE_list, hidden_dim_list, loss_list, perc_acc_list  # TODO : RMSE instead of loss; percentage acc instead of loss

def tune_model_architecture_groups(project, database, date, selected_group_list, hidden_dim, num_epochs,
                                        learning_rate, regularization_term,
                                        studied_attr, VISU=False, new_folder=False, folder=None, reference=None):
    AE_list = []  # TODO : how can this process be done with validationset and not train set?

    for selected_group in selected_group_list
        myAE = I_Autorec_model(project, database, date, selected_group, hidden_dim, num_epochs, learning_rate,
                               regularization_term,
                               studied_attr, VISU=False, new_folder=False, folder=None, reference=None)
        AE_list.append(myAE)

    """
    [mystudy, x_train, x_test, x_val, trainloader, testloader, valloader, model, tr_losses, te_losses, val_losses,
     perc_acc] = model
    """"

    loss_list = []
    perc_acc_list = []
    for AE in AE_list:
        loss_list.append(AE[9])
        perc_acc_list.append(AE[11])

    return AE_list, selected_group_list, loss_list, perc_acc_list  # TODO : RMSE instead of loss; percentage acc instead of loss

def hypertune_model_architecture_regul(project, database, date, selected_group, hidden_dim, num_epochs,
                                        learning_rate, regularization_term_list,
                                        studied_attr, VISU=False, new_folder=False, folder=None, reference=None):
    AE_list = []  # TODO : how can this process be done with validationset and not train set?

    for regularization_term in regularization_term_list
        myAE = I_Autorec_model(project, database, date, selected_group, hidden_dim, num_epochs, learning_rate,
                               regularization_term,
                               studied_attr, VISU=False, new_folder=False, folder=None, reference=None)
        AE_list.append(myAE)

    """
    [mystudy, x_train, x_test, x_val, trainloader, testloader, valloader, model, tr_losses, te_losses, val_losses,
     perc_acc] = model
    """"

    loss_list = []
    perc_acc_list = []
    for AE in AE_list:
        loss_list.append(AE[9])
        perc_acc_list.append(AE[11])

    return AE_list, regularization_term_list, loss_list, perc_acc_list  # TODO : RMSE instead of loss; percentage acc instead of loss

def plot_x_y_graph(x_list, y_list, x_label, y_label title=None, folder=None, VISU=False):
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


