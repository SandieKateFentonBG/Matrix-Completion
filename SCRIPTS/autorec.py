import numpy as np
import torch

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Autorec(nn.Module):
    def __init__(self, input_features, hidden_dim):
        super().__init__()
        self.fci = nn.Linear(input_features, hidden_dim)
        #self.fch = nn.Linear(hidden_dim, hidden_dim) #needed?
        self.fco = nn.Linear(hidden_dim, input_features)

    def forward(self, x):
        x = F.relu(self.fci(x))
        #x = F.relu(self.fch(x))
        x = self.fco(x) #TODO : identity for last layer / no reLu?