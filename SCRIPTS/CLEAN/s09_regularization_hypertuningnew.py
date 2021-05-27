
from s08_matrix_completionnew import *


def identify_tuning_parameter(param_dict):

    regul_list = param_dict['regularization_term']
    hidden_list = param_dict['hidden_dim']
    i_u_study = param_dict['i_u_study']
    database_group_split = param_dict['database_group_split']
    selected_group = param_dict['selected_group']
    batch_size = param_dict['batch_size']
    input_dim = param_dict['input_dim']
    num_epochs = param_dict['num_epochs']
    learning_rate = param_dict['learning_rate']
    device = param_dict['device']

    return [[regul_list,hidden_list],[i_u_study,database_group_split, selected_group, batch_size, input_dim, num_epochs, learning_rate, device ]]


def run_model_architectures(param_dict, project, database, date, repository,
                    VISU = True, new_folder = True):

    regul_list,hidden_list = identify_tuning_parameter(param_dict)[0]
    i_u_study, database_group_split, selected_group, batch_size, input_dim, num_epochs, learning_rate, device = \
        identify_tuning_parameter(param_dict)[1]

    AE_list = []

    for r in regul_list:
        for h in hidden_list:
            myAE = matrix_completion(i_u_study = i_u_study, database_group_split = database_group_split, selected_group = selected_group,
                         batch_size = batch_size, input_dim = input_dim, hidden_dim=h, num_epochs=num_epochs,
                         learning_rate=learning_rate, regularization_term=r, device=device, new_folder=new_folder,
                         project=project, database=database, date=date, repository=repository, VISU = VISU)

            AE_list.append(myAE)

    return AE_list

def tune_model_architectures(x0):

    from scipy.optimize import minimize

    fun = matrix_completion_function
    #x0 = [0.01, 500]
    method = 'BFGS'
    minimize(fun, x0, method=method) #todo : return what?


def matrix_completion_function(model_parameters):

    myAE = matrix_completion( hidden_dim = model_parameters[1], regularization_term = model_parameters[0])
    return - myAE.best_val_acc






