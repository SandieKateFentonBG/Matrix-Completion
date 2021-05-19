from s09_I_autorec_model import *
from s08_regularization_hypertuning import *

#Reference
project = 'Matrix-Completion'
database = 'JesterDataset4/JesterDataset4.csv'
date = "210518"

#Parameter options
selected_group_parameters = [0, 1, 2, 3, 4]
hidden_dim_parameters = [10, 20, 40, 80, 100, 200, 300, 400, 500]
num_epochs_parameters = [5, 10, 15]
learning_rate_parameters = [0.001, 0.01, 0.1, 1, 100, 1000]
regularization_term_parameters = [0.001, 0.01, 0.1, 1, 100, 1000]
studied_attr = ['selected_group','hidden_dim','num_epochs','learning_rate','regularization_term']

#TODO : train on training - validate on validation - how?
#TODO : understand how to work with 5 groups
#TODO: > average results for groups
#TODO : export results to csv
#TODO : plot graphs
#TODO : repaie default attr
#TODO : chexk sparse bp

#Default AE
default_AE = I_Autorec_model(project = project, database = database, date = date, selected_group=selected_group_parameters[0],
                             hidden_dim=hidden_dim_parameters[8], num_epochs=num_epochs_parameters[0],
                             learning_rate=learning_rate_parameters[1], regularization_term=regularization_term_parameters[0],
                             studied_attr = 'selected_group', VISU = True, new_folder=True, folder = True)

# myAE = [mystudy = 0, x_train = 1, x_test = 2, x_val = 3, trainloader = 4, testloader = 5, valloader = 6, model = 7,
# tr_losses = 8, te_losses = 9, rmse_losses = 10, best_val_loss = 11, perc_acc = 12]


#AE on all groups
AE_list, selected_group_list, autorec_loss_list, perc_acc_list, rmse_loss_list, best_val_loss_list\
    = tune_model_architecture_groups(project = project, database = database,
                            date = date, selected_group_list=selected_group_parameters,
                             hidden_dim=hidden_dim_parameters[8], num_epochs=num_epochs_parameters[0],
                             learning_rate=learning_rate_parameters[1], regularization_term=regularization_term_parameters[0],
                             studied_attr = studied_attr[0], VISU = True, new_folder=False, folder = True)

#hypertune for variable regularization strength
AE_list, regularization_term_list, autorec_loss_list, perc_acc_list, rmse_loss_list, best_val_loss_list  = \
    hypertune_model_architecture_regul (project = project,
                                database = database, date = date, selected_group=selected_group_parameters[0],
                             hidden_dim=hidden_dim_parameters[8], num_epochs=num_epochs_parameters[0],
                             learning_rate=learning_rate_parameters[1], regularization_term_list = regularization_term_parameters,
                             studied_attr = studied_attr[4], VISU = True, new_folder=False, folder = True)

#hypertune for variable regularization strength
AE_list, hidden_dim_list, autorec_loss_list, perc_acc_list, rmse_loss_list, best_val_loss_list =\
    tune_model_architecture_hidden(project = project,
                                   database = database, date = date, selected_group=selected_group_parameters[0],
                                   hidden_dim_list=hidden_dim_parameters, num_epochs=num_epochs_parameters[0],
                                   learning_rate=learning_rate_parameters[1], regularization_term=regularization_term_parameters[0],
                                   studied_attr = studied_attr[1], VISU = True, new_folder=False, folder = True)

#plot
#plot_x_y_graph(reg_list, loss_list, x_label = 'regularization_term', y_label ='autorec_loss', title=None, folder=None, VISU=False)
#plot_sns_graph(reg_list, loss_list, x_label= 'regularization_term', y_label ='autorec_loss', title=None, figure_size=(12,15), folder=None, plot=False)
