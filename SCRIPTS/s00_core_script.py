from s10_I_autorec_model import *
from s08_regularization_hypertuning import *

#Reference
project = 'Matrix-Completion'
database = 'JesterDataset4/JesterDataset4.csv'
date = "210518"

#Parameter options
selected_group_parameters = [0, 1, 2, 3, 4] # TODO : do it on all then keep only best results?
hidden_dim_parameters = [10, 20, 40, 80, 100, 200, 300, 400, 500]
num_epochs_parameters = [5, 10, 15] # TODO : choice?
learning_rate_parameters = [0.001, 0.01, 0.1, 1, 100, 1000] # TODO : choice?
regularization_term_parameters = [0.001, 0.01, 0.1, 1, 100, 1000]
studied_attr = ['selected_group','hidden_dim','num_epochs','learning_rate','regularization_term', 'Default_settings']

#Default AE
default_AE = I_Autorec_model(project = project, database = database, date = date, selected_group=selected_group_parameters[0],
                             hidden_dim=hidden_dim_parameters[8], num_epochs=num_epochs_parameters[0],
                             learning_rate=learning_rate_parameters[1], regularization_term=regularization_term_parameters[0],
                             studied_attr = studied_attr[5], VISU = True, new_folder=False, folder = True)

# myAE = [mystudy = 0, x_train = 1, x_test = 2, x_val = 3, trainloader = 4, testloader = 5, valloader = 6, model = 7,
# tr_losses = 8, te_losses = 9, val_losses = 10, perc_acc = 11]

#AE on all groups
AE_list, gr_list, loss_list, acc_list = tune_model_architecture_groups(project = project, database = database,
                            date = date, selected_group_list=selected_group_parameters,
                             hidden_dim=hidden_dim_parameters[8], num_epochs=num_epochs_parameters[0],
                             learning_rate=learning_rate_parameters[1], regularization_term=regularization_term_parameters[0],
                             studied_attr = studied_attr[0], VISU = True, new_folder=False, folder = True)


#hypertune for constant architecture/fixed AE weights

reg_list, loss_list, perc_acc = hypertune_model_parameters_regularization(default_AE, regularization_term_parameters)
test_print = print('CHECK - should all be the same', perc_acc)
plot_x_y_graph(reg_list, loss_list, x_label = 'regularization_term', y_label ='autorec_loss', title=None, folder=None, VISU=False)
plot_sns_graph(reg_list, loss_list, x_label= 'regularization_term', y_label ='autorec_loss', title=None, figure_size=(12,15), folder=None, plot=False)

#hypertune for variable architecture/variable AE weights
#
# TODO : regularization strength only influences the loss value
#  > should not be hypertuned on a fix weight matrix but should be used to optimize weights
#  #check that loss = list for 5 epochs and not the overall best.. ?

AE_list, regularization_term_list, loss_list, perc_acc_lis = hypertune_model_architecture_regul (project = project,
                                database = database, date = date, selected_group=selected_group_parameters[0],
                             hidden_dim=hidden_dim_parameters[8], num_epochs=num_epochs_parameters[0],
                             learning_rate=learning_rate_parameters[1], regularization_term_list = regularization_term_parameters,
                             studied_attr = studied_attr[4], VISU = True, new_folder=False, folder = True)

AE_list, hidden_dim_list, loss_list, perc_acc_list = hypertune_model_architecture_hidden(project = project,
                            database = database, date = date, selected_group=selected_group_parameters[0],
                             hidden_dim_list=hidden_dim_parameters, num_epochs=num_epochs_parameters[0],
                             learning_rate=learning_rate_parameters[1], regularization_term=regularization_term_parameters[0],
                             studied_attr = studied_attr[1], VISU = True, new_folder=False, folder = True)