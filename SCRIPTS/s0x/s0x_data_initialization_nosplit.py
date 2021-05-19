
import numpy as np
import torch

def load_data(path = 'C:/Users/sfenton/Code/Repositories/Matrix-Completion/DATA/JesterDataset4/JesterDataset4.csv', study = 0):
    # Load data
    user_data = np.genfromtxt(path, delimiter=';', dtype='str',
                              usecols=(i for i in range(1, 159)))  # first column removed
    user_data = np.char.replace(user_data, ',', '.').astype(float)
    item_data = np.transpose(user_data)
    user_tensor = torch.tensor(user_data)
    item_tensor = torch.tensor(item_data)
    combined_data = [item_data, user_data]
    return combined_data[study]

def split_data(count, studied_data):
    N_split = int((1/count) * studied_data.shape[1])
    list = []
    for i in range(1, count):
        list.append(N_split*i)
        print(list)
    print(studied_data.shape[0]-(N_split*(count-1)))
    print(studied_data.shape[0])
    list.append(studied_data.shape[0]-(N_split*(count-1)))

    return torch.utils.data.random_split(studied_data,list)


def split_train_test_validate_data(studied_data):
    # split into training (81%), validation(9%) and test sets(10%)   #TODO: repeat 5 times? shuffle? mean/scale/normalize?
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