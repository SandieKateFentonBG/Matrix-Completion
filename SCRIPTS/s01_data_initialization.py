
import numpy as np
import torch

def load_data(path = 'C:/Users/sfenton/Code/Repositories/Matrix-Completion/DATA/JesterDataset4/JesterDataset4.csv', study):
    # Load data
    user_data = np.genfromtxt(path, delimiter=';', dtype='str',
                              usecols=(i for i in range(1, 159)))  # first column removed
    user_data = np.char.replace(user_data, ',', '.').astype(float)
    item_data = np.transpose(user_data)
    user_tensor = torch.tensor(user_data)
    item_tensor = torch.tensor(item_data)  # TODO: convert 99s?
    combined_data = [item_data, user_data]
    return combined_data[study]


def split_data(studied_data):
    # split into training (81%), validation(9%) and test sets(10%)   #TODO: repeat 5 times? shuffle? mean/scale/normalize?
    N_split = int(0.9 * studied_data.shape[0])
    x_temp = studied_data[:N_split, :]
    x_test = studied_data[N_split:, :]
    N_split2 = int(0.9 * x_temp.shape[0])
    x_train = x_temp[:N_split2, :]
    x_val = x_temp[N_split2:, :]
    return x_train, x_test, x_val

def construct_tensor(array):
    return torch.tensor(array)

def create_data_loader(x_train, x_test, x_val):
    # Create dataloaders from tensors #TODO: shuffle?
    trainloader = torch.utils.data.DataLoader(construct_tensor(x_train), batch_size=batch_size, shuffle=True)
    testloader = torch.utils.data.DataLoader(construct_tensor(x_test), batch_size=batch_size, shuffle=False)
    valloader = torch.utils.data.DataLoader(construct_tensor(x_val), batch_size=batch_size, shuffle=False)
    return trainloader, testloader, valloader