from s01_parameter_selection import *
from s02_data_initialization import *
from s03_model_definition import *
from s06_training_loop import *
from s07_investigate_results import *

def I_Autorec_model(project, database, date, selected_group, hidden_dim, num_epochs, learning_rate, regularization_term,
                    studied_attr = None, val = 'Default', VISU = False, new_folder=False, folder = None, reference = None):

    # PARAM SELECTION
    mystudy = case_study(selected_group=selected_group, hidden_dim=hidden_dim, num_epochs=num_epochs,
                         learning_rate=learning_rate, regularization_term=regularization_term)
    if studied_attribute:
        val = mystudy.__getattribute__(studied_attr)
    ref, input_path, output_path = input_study_display(mystudy, date, project, database, studied_attr, val,
                                                       folder = folder, new_folder=new_folder, VISU = VISU)

    if folder :
        folder = output_path
        reference = ref

    # FORMAT DATA
    user_data = load_data(input_path)
    sections = split_data(mystudy.database_group_split, user_data)
    test_section = sections[mystudy.selected_group]
    studied_data = selected_data(test_section, mystudy.study)
    x_train, x_test, x_val = split_train_test_validate_data(studied_data)
    trainloader, testloader, valloader = create_data_loader(x_train, x_test, x_val, mystudy.batch_size)
    data_initialization_print(x_train, x_test, x_val, folder=folder, VISU = VISU, reference = reference)

    # INSTANTIATE MODEL
    size = x_train.shape[1]
    model = Autorec(size, mystudy.hidden_dim).to(mystudy.device)
    optimizer = torch.optim.Rprop(model.parameters(), mystudy.learning_rate)
    model_definition_print(model, optimizer, folder=folder, VISU = VISU, reference = reference)
    evaluation_function_print(model, loss_function='custom autorec loss', folder=folder, VISU = VISU, reference = reference)

    #TRAIN MODEL
    tr_losses, te_losses, rmse_losses, best_val_loss = training_loop(model, optimizer, mystudy.num_epochs, trainloader, testloader,
                                                     mystudy.device, mystudy.regularization_term, folder=folder,
                                                     VISU = VISU, reference = reference)

    #EVALUATE RESULTS
    perc_acc = compute_perc_acc(testloader, mystudy.device, model, round=False, folder=folder, VISU = VISU, reference = reference)
    plot_results(tr_losses, te_losses, reference, folder, VISU = VISU)


    #TODO: set model results in class
    model.rmse_loss =

    #TODO: set model results in class

    return [mystudy, x_train, x_test, x_val, trainloader, testloader, valloader, model, tr_losses, te_losses, rmse_losses, best_val_loss, perc_acc]
