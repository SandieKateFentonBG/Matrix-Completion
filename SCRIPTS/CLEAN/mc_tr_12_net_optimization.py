
from mc_tr_11_initial_training import *


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

def mc_training_calibration(data_dict, mydata, default_params, model, optimizer, row_variable = regul_list,
                            column_variable = hidden_list, row_label = 'regul_term', column_label = 'hidden_dim',
                            threshold = None, score = 'rmse_te_losses'):

    results_matrix = np.zeros(len(row_variable), len(column_variable))
    score_matrix = np.zeros(len(row_variable), len(column_variable))
    for r in range(len(row_variable)):
        default_params.regularization_term = row_variable[r]
        for h in range(len(column_variable)):
            default_params.hidden_dim = column_variable[h]
            reference = model_reference(selected_group=mydata.selected_group, hidden_dim=column_variable[h],
                                        regularization_term=row_variable[r], extra=threshold)

            # CALIBRATE MODEL
            loss_evolution_dict, best_score_dict = net_calibration(model, optimizer, mydata, default_params, reference, data_dict['trainloader'],
                                        data_dict['testloader'], VISU=VISU, threshold=threshold)

            training_calibration_print(row_label, row_variable[r], column_label, column_variable[h], reference,
                                           score_dict=best_score_dict, output_path=mydata.output_path, VISU=VISU)

            results_matrix[r][h] = best_score_dict
            score_matrix[r][h] = best_score_dict[score]

    return {'results_matrix': results_matrix, 'score_matrix': score_matrix, 'myparams': default_params, 'model' : model, 'optimizer': optimizer, 'loss_dict': loss_dict}


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






