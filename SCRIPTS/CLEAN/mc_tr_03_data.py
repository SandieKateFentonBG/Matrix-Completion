from mc_tr_01_data_parameters import *
from mc_tr_02_data_preprocessing import *

def data_initialization(mystudy):

    mydata = load_data(mystudy.input_path)
    sections = split_data(mystudy.database_group_split, mydata)
    test_section = sections[mystudy.selected_group]
    studied_data = selected_data(test_section, mystudy.i_u_study)
    x_train, x_test, x_val = split_train_test_validate_data(studied_data)
    trainloader, testloader, valloader = create_data_loader(x_train, x_test, x_val, mystudy.batch_size)

    data_dict = dict()
    data_dict['x_train'] = x_train
    data_dict['x_test'] = x_test
    data_dict['x_val'] = x_val
    data_dict['trainloader'] = trainloader
    data_dict['testloader'] = testloader
    data_dict['valloader'] = valloader

    return data_dict

def data_initialization_print(mystudy, reference, data_dict, folder=None, new_folder=False, VISU=False):

    # Print data
    if new_folder:
        from mc_tr_05_Autorec_net import mkdir_p
        mkdir_p(folder)
    keys = ['x_train', 'x_test', 'x_val']
    for k in keys:
        #v in data_dict.items():
        if VISU:
            print(k, data_dict[k].shape)
    if folder:
        with open(folder + reference + ".txt", 'a') as f:
            print(k,data_dict[k].shape, file=f)
        f.close()

def sanity_check(dataloader):
    # Get random training ratings
    dataiter = iter(dataloader)
    example_ratings = dataiter.next()
    return example_ratings

def mc_tr_data(project, database, date, repository,
            i_u_study = 0, database_group_split = 5, selected_group = 0, batch_size = 8,input_dim = [7699, 158],
                    VISU = True, new_folder=False):

    # CASE STUDY INITIALIZATION
    mydata = Training_data(i_u_study = i_u_study, database_group_split = database_group_split,
                           selected_group = selected_group, batch_size = batch_size, input_dim = input_dim,
                         project=project, database=database, date=date, repository=repository)

    mc_tr_data_print(mystudy, folder=mystudy.output_path, new_folder = new_folder, VISU=VISU)

    # FORMAT DATA
    data_dict = data_initialization(mydata)
    data_initialization_print(mydata, data_dict, folder=mydata.output_path, VISU=VISU)

    return data_dict

