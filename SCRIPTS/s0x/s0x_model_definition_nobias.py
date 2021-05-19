import numpy as np
import torch

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Autorec(nn.Module):
    def __init__(self, input_features, hidden_dim):
        super().__init__()
         #TODO : bias = True - see how to deal with this in s04 evaluation function - autorec loss
        #        # here you could add Vbias, Wbias through a reshape = model.fci.bias, model.fco.bias
        self.fci = nn.Linear(input_features, hidden_dim, bias = False) #TODO: here bias change
        #self.fch = nn.Linear(hidden_dim, hidden_dim) #needed?
        self.fco = nn.Linear(hidden_dim, input_features, bias = False) #TODO: here bias change

    def forward(self, x):
        temp = self.fci(x)
        x = F.relu(temp)
        #x = F.relu(self.fch(x))
        x = self.fco(x) #TODO : identity for last layer / no reLu?
        return x