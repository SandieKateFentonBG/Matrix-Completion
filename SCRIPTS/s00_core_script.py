
import numpy as np
import torch

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Parameters

study = 0 # 0=item_data, 1=user_data
batch_size = 8
input_dim = [7699,158] # ratings_per_item, ratings_per_user
hidden_dim_count = 500 #{10, 20, 40, 80, 100, 200, 300, 400, 500}
num_epochs = 5 # cfr repeat process 5 times
learning_rate = 0.001 #{0.001, 0.01, 0.1, 1, 100, 1000}
device = 'cuda:0'
regularization_term = 0.5
input_path = 'C:/Users/sfenton/Code/Repositories/Matrix-Completion/DATA/JesterDataset4/JesterDataset4.csv'

"""
1. Parameter initialization and data preparation
"""
from s01_data_initialization import *

# Load data
studied_data = load_data(path = input_path, study)
x_train, x_test, x_val = split_data(studied_data)
trainloader, testloader, valloader = create_data_loader(x_train, x_test, x_val)

# TODO :Questions
#  1. repeat 5 times? shuffle? mean/scale/normalize?
#  2. at what stage should we do dataloader vs full data > batch size dim?

# Print data
print("train :", x_train.shape)
print("test :", x_test.shape)
print("validate :", x_val.shape)

"""
2. Sanity check  
"""
from s02_sanity_check import *

# Print output
check1 = sanity_check1(trainloader)

# TODO :Questions
#  1. check later

"""
3. Model definition
"""
from s03_model_definition import *

# Instantiate model and optimizer
model = Autorec(input_dim[study], hidden_dim_count).to(device)
weights = torch.optim.Rprop(model.parameters(), learning_rate)
"""
optimizer/weights
    params (iterable): iterable of parameters to optimize or dicts defining
        parameter groups
    lr (float, optional): learning rate (default: 1e-2)
    etas (Tuple[float, float], optional): pair of (etaminus, etaplis), that #TODO:                  
        are multiplicative increase and decrease factors
        (default: (0.5, 1.2))
    step_sizes (Tuple[float, float], optional): a pair of minimal and
        maximal allowed step sizes (default: (1e-6, 50))
#'Rprop does not support sparse gradients'
#https://pytorch.org/docs/master/_modules/torch/optim/rprop.html#Rprop
"""

# Print output
print('model', model)

# TODO :Questions
#  1. wat difference if model is a class?
#  2. model - identify for last layer / no reLu?
#  3. optimizer vs weights?
#  4. default weights ? What is this etas?

"""
4. Evaluation functions
"""
from s04_evaluation_functions import *

# Print output
print("Number of trainable parameters: {}".format(count_parameters(model)))

# TODO :Questions
#  1.Number of trainable parameters: 7707199???
#  2.loss vs objective? check autorec_loss content

"""
5. Training Loop
"""
from s05_running_loss import *

# TODO : Questions:
#  1. sparsity ?

"""
6. Training Loop
"""

from s06_training_loop import *

tr_losses, te_losses = training_loop(model,weights, num_epochs, trainloader, testloader)

# TODO : Questions:
#  1. only update the weights with observed inputs / although we compute them for everything ?

"""
7. Investigate results
"""
from s07_plot_results import *

plot_results(tr_losses,te_losses)
#Check some reconstructions by the trained model
check2 = sanity_check2(testloader)

#TODO: check this later

# Show images
preds = model(example_rating.to(device))
print(' '.join('%5s' % example_ratings[j].item() for j in range(batch_size)))
# Print labels
print(' '.join('%5s' % example_labels[j].item() for j in range(batch_size)))

"""
7. Parameter hypertuning - with validation set
"""
# tune params with with validation set
# test deep AE