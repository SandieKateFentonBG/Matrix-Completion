import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision

#https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module
# Load dataset

data = torch.utils.data.Dataset


transform = torchvision.transforms.Compose(
    [torchvision.transforms.ToTensor(),
     torchvision.transforms.Normalize((0.5), (0.5))])

trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                        download=True, transform=transform)
valset = torchvision.datasets.MNIST(root='./data', train=False,
                                       download=True, transform=transform)

print(trainset)

# Create dataloaders
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True)
valloader = torch.utils.data.DataLoader(valset, batch_size=batch_size,
                                         shuffle=False)
"""
def open_csv_at_given_line(filename, first_line=0, delimiter=';'):
    import csv
    reader = csv.reader(open(filename + '.csv', mode='r'), delimiter=delimiter)
    for i in range(first_line):
        reader.__next__()
    header = reader.__next__()
    return header, reader

def load_ratings_per_user(filename, first_line=0, delimiter=';'):
    import csv
    reader = csv.reader(open(filename + '.csv', mode='r'), delimiter=delimiter)
    for i in range(first_line):
        reader.__next__()
    header = reader.__next__()
    return header

def load_ratings_per_item(filename, first_line=0, delimiter=';'):
    import csv
    reader = csv.reader(open(filename + '.csv', mode='r'), delimiter=delimiter)
    item_ratings = []
    for i in range(len(reader)):
        item_rating = []
        item_rating.append(reader[i])

        reader.__next__()
    header = reader.__next__()
    return header, reader

path = 'C:/Users/sfenton/Code/Repositories/Matrix-Completion/DATA/JesterDataset4/JesterDataset4'
user_rating_a = open_csv_at_given_line(path)

class AutoencoderDense(nn.Module):
    def __init__(self, input_features, hidden_dim):
        super().__init__()
        self.tensor = nn.Tensor() #creates a tensor
        self.fci = nn.Linear(input_features, hidden_dim)
        #self.fch = nn.Linear(hidden_dim, hidden_dim) #needed?
        self.fco = nn.Linear(128, 128)

    def forward(self, x):
        x = F.relu(self.fci(x))
        #x = F.relu(self.fch(x))
        x = F.relu(self.fco(x))
        x = x.view(-1, 1, 28, 28)
        #view: reshape this tensor from -1*1 to
        #-1 : use when unknown number of rows/cols

"""
"""print(a)
print(type(a))
print(len(a))
print(a[0])
print(type(a[0]))
print(len(a[0]))
print(a[1])"""
