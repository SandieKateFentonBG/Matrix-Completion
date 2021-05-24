from s01_parameter_selectionnew import *
from s02_data_initializationnew import *
from s03_model_definitionnew import *
from s06_training_loopnew import *
from s07_model_resultsnew import *

def matrix_completion(project, database, date, repository,
            i_u_study = 0, database_group_split = 5, selected_group = 0, batch_size = 8, input_dim = [7699, 158],
                hidden_dim = 500, num_epochs = 5, learning_rate = 0.001, regularization_term = 0.01, device = 'cuda:0',
                    VISU = True, new_folder=False):

    # CASE STUDY INITIALIZATION
    mystudy = case_study(i_u_study = i_u_study, database_group_split = database_group_split, selected_group = selected_group,
                         batch_size = batch_size, input_dim = input_dim, hidden_dim=hidden_dim, num_epochs=num_epochs,
                         learning_rate=learning_rate, regularization_term=regularization_term, device=device,
                         project=project, database=database, date=date, repository=repository)

    case_study_print(mystudy, folder=mystudy.output_path, new_folder = new_folder, VISU=VISU)

    # FORMAT DATA
    data_dict = data_initialization(mystudy)
    data_initialization_print(mystudy, data_dict, folder=mystudy.output_path, VISU=VISU)

    # INSTANTIATE MODEL
    model = Autorec(data_dict['x_train'].shape[1], mystudy.hidden_dim).to(mystudy.device)
    optimizer = torch.optim.Rprop(model.parameters(), mystudy.learning_rate)
    model_definition_print(model, optimizer, folder=mystudy.output_path, VISU = VISU, reference = mystudy.reference)

    #TRAIN MODEL
    loop_dict = training_loop(model, optimizer, mystudy, data_dict['trainloader'], data_dict['testloader'],VISU = VISU)

    #STORE RESULTS
    matrix_results = model_results(model, optimizer, mystudy, loop_dict['tr_losses'], loop_dict['te_losses'], loop_dict['rmse_te_losses'],
                         loop_dict['te_accuracies'], loop_dict['best_val_acc'], loop_dict['tr_accuracies'])
    model_results_print(matrix_results, folder=mystudy.output_path, new_folder=False, VISU=VISU)


    return matrix_results

