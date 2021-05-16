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
        self.hidden = nn.Linear(input_features, hidden_dim)
        self.predict = nn.Linear(hidden_dim, input_features)

    def forward(self, x):
        temp = self.hidden(x)
        x = F.relu(temp)
        x = self.predict(x) #TODO : identity for last layer / no reLu?
        return x