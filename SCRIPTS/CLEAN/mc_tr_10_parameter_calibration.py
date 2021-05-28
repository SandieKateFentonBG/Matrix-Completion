import numpy as np
from mc_tr_05_Autorec_net import *
from mc_tr_08_net_calibration import *

def mc_training_calibration(data_dict, mydata, default_params, model, optimizer, row_variable,
                            column_variable, row_label = 'regul_term', column_label = 'hidden_dim',
                            threshold = None, score = 'rmse_te_losses', VISU = False, new_folder=False):

    results_matrix = np.zeros((len(row_variable), len(column_variable)))
    score_matrix = np.zeros((len(row_variable), len(column_variable)))
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

            #results_matrix[r][h] = best_score_dict
            results_matrix[r][h] = best_score_dict['te_losses']
            score_matrix[r][h] = best_score_dict[score]

    return {'results_matrix': results_matrix, 'score_matrix': score_matrix, 'myparams': default_params, 'model' : model, 'optimizer': optimizer}


def training_calibration_print(row_label, row_value, col_label, col_value, reference, score_dict, output_path=None, VISU=False):

    for k, v in score_dict.items():
        if VISU:
            print(row_label, ' : ', row_value, col_label, ' : ', col_value)
            print(' ', k, ' : ', v)
        if output_path:
            with open(output_path + reference + ".txt", 'a') as f:
                print(row_label, ' : ', row_value, col_label, ' : ', col_value, file=f)
                print(' ', k, ' : ', v, file=f)
            f.close()






