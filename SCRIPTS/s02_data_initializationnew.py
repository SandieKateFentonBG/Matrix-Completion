
import numpy as np
import torch

def load_data(path = 'C:/Users/sfenton/Code/Repositories/Matrix-Completion/DATA/JesterDataset4/JesterDataset4.csv', ):
    # Load data
    user_data = np.genfromtxt(path, delimiter=';', dtype='str',
                              usecols=(i for i in range(1, 159)))  # first column removed
    user_data = np.char.replace(user_data, ',', '.').astype(float)
    return user_data

def split_data(count, user_data, split = 0):

    N_split = int((1/count) * user_data.shape[split])
    list = [N_split]
    for i in range(1, count-1):
        list.append((N_split*(i+1))-(N_split*i))
    list.append(user_data.shape[split] - (N_split * (count - 1)))

    return torch.utils.data.random_split(user_data, list)

def selected_data(user_data, study) :
    item_data = np.transpose(user_data)
    combined_data = [item_data, user_data]
    return combined_data[study]

def split_train_test_validate_data(studied_data):
    # split into training (81%), validation(9%) and test sets(10%)   #TODO:  mean/scale/normalize?
    N_split = int(0.9 * studied_data.shape[0])
    x_temp = studied_data[:N_split, :]
    x_test = studied_data[N_split:, :]
    N_split2 = int(0.9 * x_temp.shape[0])
    x_train = x_temp[:N_split2, :]
    x_val = x_temp[N_split2:, :]
    return x_train, x_test, x_val

def construct_tensor(array):
    return torch.tensor(array).float()

def create_data_loader(x_train, x_test, x_val, batch_size):
    # Create dataloaders from tensors
    trainloader = torch.utils.data.DataLoader(construct_tensor(x_train), batch_size=batch_size, shuffle=True) #TODO : shuffle = true > my results differ every time?
    testloader = torch.utils.data.DataLoader(construct_tensor(x_test), batch_size=batch_size, shuffle=False)
    valloader = torch.utils.data.DataLoader(construct_tensor(x_val), batch_size=batch_size, shuffle=False)
    return trainloader, testloader, valloader

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

def data_initialization_print(mystudy, data_dict, folder=None, new_folder=False, VISU=False):

    from s10_export_resultsnew import mkdir_p
    # Print data
    if new_folder:
        mkdir_p(folder)
    keys = ['x_train', 'x_test', 'x_val']
    for k, v in data_dict.items():
        if VISU:
            print(k, v.shape)
    if folder:
        with open(folder + mystudy.reference + ".txt", 'a') as f:
            print(k, v.shape)
        f.close()

def sanity_check(dataloader):
    # Get random training ratings
    dataiter = iter(dataloader)
    example_ratings = dataiter.next()
    return example_ratings


