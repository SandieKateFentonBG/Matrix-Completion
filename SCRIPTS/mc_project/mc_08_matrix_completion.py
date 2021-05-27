
from SCRIPTS.mc_project.mc_01_parameters_initialization import *
from mc_07_net_calibration import *
from mc_03_net_architecture import *
from mc_05_evaluation_criteria import *


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

    # CALIBRATE MODEL
    loop_dict = net_calibration(model, optimizer, mystudy, data_dict['trainloader'], data_dict['testloader'],VISU = VISU)
    net_calibration_print(loop_dict, output_path=mystudy.output_path, reference=mystudy.reference, VISU=VISU)

    #STORE RESULTS



    return matrix_results


