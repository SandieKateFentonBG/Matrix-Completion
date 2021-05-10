import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import csv
import io

import pandas as pd
path = 'C:/Users/sfenton/Code/Repositories/Matrix-Completion/DATA/JesterDataset4/JesterDataset4.csv'

#with open (path, "r") as myfile:
#       data=myfile.read().replace('"', '')
#np.genfromtxt(io.cStringIO.StringIO(data), skip_header=1, delimiter=",", names = True)


df = pd.read_csv(path, delimiter=';')
df.values
user_data = np.genfromtxt(path, delimiter=';', dtype='str') #usecols=[1:-1]
user_data = np.char.replace(user_data, ',', '.').astype(float)
item_data = np.transpose(user_data)

user_tensor = torch.tensor(user_data) #'pandas.core.frame.DataFrame'
item_tensor = torch.tensor(item_data)

print(len(user_tensor[0]))
print(len(item_tensor[0]))
print(item_tensor[7])
print(user_tensor[0])
#print(user_tensor[0])
#for elem in item_tensor[7]:
#    print(elem, type(elem))


#print(user_data)


#reader = csv.reader(open(path + '.csv', mode='r'), delimiter=';')
#data = torch.tensor(reader)
#data = torch.utils.data.Dataset
#print(type(reader))