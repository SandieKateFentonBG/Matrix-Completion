import numpy as np
import torch

def loadData(path):
    user_data = np.genfromtxt(path, delimiter=';', dtype='str', usecols=(i for i in range(1, 159)))
    return np.char.replace(user_data, ',', '.').astype(float)

def makeSubdata(numberOfSubdata, data, dimensionToSplit = 0):
    subdataSize = int(data.shape[dimensionToSplit] / numberOfSubdata)
    actualSubdataSizes = [subdataSize for i in range(numberOfSubdata - 1)]
    actualSubdataSizes.append(data.shape[dimensionToSplit] - sum(actualSubdataSizes))
    return torch.utils.data.random_split(data, actualSubdataSizes)

def selectData(user_data, itemBased):
    if itemBased:
        return np.transpose(user_data)
    return user_data

# split into training (81%), validation(9%) and test sets(10%)
def makeLearningsGroups(studied_data):
    trainingStopIndex = int(0.81 * studied_data.shape[0])
    valStopIndex = trainingStopIndex + int(0.09 * studied_data.shape[0])
    return [studied_data[:trainingStopIndex], studied_data[trainingStopIndex:valStopIndex], studied_data[valStopIndex:]]

def construct_tensor(array):
    return torch.tensor(array).float()

def makeDataLoader(x_train, x_test, x_val, batch_size):
    return [torch.utils.data.DataLoader(construct_tensor(data), batch_size=batch_size, shuffle=data is x_train)
            for data in [x_train, x_test, x_val]]
