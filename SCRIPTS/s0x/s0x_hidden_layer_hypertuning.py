from s01_parameter_selection import *
from s02_data_initialization import *
from s03_model_definition import *
from s06_training_loop import *
from s07_investigate_results import *

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

def I_Autorec_model(project = project, database = database, date = date, selected_group=selected_group_parameters[0],
                    hidden_dim=hidden_dim_parameters[8], num_epochs=num_epochs_parameters[0],
                    learning_rate=learning_rate_parameters[1], regularization_term=regularization_term_parameters[0],
                    VISU = False, new_folder=False, folder = None):

    # PARAM SELECTION
    mystudy = case_study(selected_group=selected_group, hidden_dim=hidden_dim,
                         num_epochs=num_epochs, learning_rate=learning_rate,
                         regularization_term=regularization_term)
    attr = 'selected_group'  # 'hidden_dim', 'num_epochs', 'learning_rate', 'regularization_term'
    val = mystudy.__getattribute__(attr)
    reference, input_path, output_path = input_study_display(mystudy, date, attr, val,
                                                             project, database, folder = folder, new_folder=new_folder, VISU = VISU)

    if folder :
        folder = output_path

    # FORMAT DATA
    user_data = load_data(input_path)
    sections = split_data(mystudy.database_group_split, user_data)
    test_section = sections[mystudy.selected_group]
    studied_data = selected_data(test_section, mystudy.study)
    x_train, x_test, x_val = split_train_test_validate_data(studied_data)
    trainloader, testloader, valloader = create_data_loader(x_train, x_test, x_val, mystudy.batch_size)
    data_initialization_print(x_train, x_test, x_val, folder=folder, VISU = VISU)

    # INSTANTIATE MODEL
    size = x_train.shape[1]
    model = Autorec(size, mystudy.hidden_dim).to(mystudy.device)
    optimizer = torch.optim.Rprop(model.parameters(), mystudy.learning_rate)
    model_definition_print(model, optimizer, folder=folder, VISU = VISU)
    evaluation_function_print(model, loss_function='custom autorec loss', folder=folder, VISU = VISU)

    #TRAIN MODEL
    tr_losses, te_losses, val_losses = training_loop(model, optimizer, mystudy.num_epochs, trainloader, testloader, mystudy.device,
                                         mystudy.regularization_term, folder=folder, VISU = VISU)

    #EVALUATE RESULTS
    perc_acc = compute_perc_acc(testloader, mystudy.device, model, round=False, folder=folder, VISU = VISU)
    plot_results(tr_losses, te_losses, reference, folder, VISU = VISU)

    return mystudy, x_train, x_test, x_val, trainloader, testloader, valloader, model, tr_losses, te_losses, val_losses, perc_acc

myAE = I_Autorec_model(project = project, database = database, date = date, selected_group=selected_group_parameters[0],
                    hidden_dim=hidden_dim_parameters[8], num_epochs=num_epochs_parameters[0],
                    learning_rate=learning_rate_parameters[1], regularization_term=regularization_term_parameters[0],
                    VISU = True, new_folder=True, folder = True)