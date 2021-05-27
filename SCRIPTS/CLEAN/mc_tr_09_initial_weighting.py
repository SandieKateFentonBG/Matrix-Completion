

from mc_tr_01_data_parameters import *
from mc_tr_02_data_preprocessing import *
from mc_tr_03_data import *
from mc_tr_04_Autorec_parameters import *
from mc_tr_05_Autorec_net import *
from mc_tr_06_evaluation_criteria import *
from mc_tr_07_loss_functions import *
from mc_tr_08_net_calibration import *

def mc_training_set_up(project, database, date, repository,
                       i_u_study = 0, database_group_split = 5, selected_group = 0, batch_size = 8, input_dim = [7699, 158],
                       hidden_dim = 500, num_epochs = 5, learning_rate = 0.001, regularization_term = 0.01, device = 'cuda:0',
                       VISU = True, new_folder=False, threshold = None):

    # STUDY REFERENCE
    reference = model_reference(selected_group = selected_group, hidden_dim = hidden_dim,
                                regularization_term = regularization_term, extra = threshold)

    # TRAINING DATA
    mydata = Training_data(i_u_study = i_u_study, database_group_split = database_group_split, selected_group = selected_group,
                         batch_size = batch_size, input_dim = input_dim,
                         project=project, database=database, date=date, repository=repository)
    mc_tr_data_print(mydata, reference, folder=mydata.output_path, new_folder = new_folder, VISU=VISU)

    data_dict = data_initialization(mydata)
    data_initialization_print(mydata, reference, data_dict, folder=mydata.output_path, VISU=VISU)

    # TRAINING PARAMETERS
    myparams = Autorec_parameters( hidden_dim=hidden_dim, num_epochs=num_epochs,
                         learning_rate=learning_rate, regularization_term=regularization_term, device=device)
    mc_tr_params_print(myparams, mydata, reference, folder=mydata.output_path, new_folder = new_folder, VISU=VISU)

    # INSTANTIATE MODEL
    print('check data_dict[x_train].shape[1]', data_dict['x_train'].shape[1]) #todo
    model = Autorec_net(data_dict['x_train'].shape[1], myparams.hidden_dim).to(myparams.device)
    optimizer = torch.optim.Rprop(model.parameters(), myparams.learning_rate)
    model_definition_print(model, optimizer, folder=mydata.output_path, VISU = VISU, reference = reference)

    # CALIBRATE MODEL
    loss_evolution_dict, best_score_dict = net_calibration(model, optimizer, mydata, myparams, reference, data_dict['trainloader'],
                                data_dict['testloader'],VISU = VISU, threshold = threshold)
    loss_calibration_print(loss_evolution_dict, reference, output_path=mydata.output_path, VISU=VISU)
    loss_calibration_print(best_score_dict, reference, output_path=mydata.output_path, VISU=VISU)

    #STORE RESULTS

    return {'data_dict': data_dict, 'mydata': mydata, 'myparams': myparams,'model' : model, 'optimizer': optimizer,
             'loss_evolution_dict':loss_evolution_dict,'best_score_dict': best_score_dict}


