
import numpy as np
import torch

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Parameters

study = 0 # 0=item_data, 1=user_data
batch_size = 8
group_split = 5
tested_section = 0
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
user_data = load_data(input_path)
sections = split_data(group_split, user_data)
test_section = sections[tested_section]
studied_data = selected_data(test_section, study = study)

x_train, x_test, x_val = split_train_test_validate_data(studied_data)

trainloader, testloader, valloader = create_data_loader(x_train, x_test, x_val, batch_size)

# TODO :Questions
#  1. repeat 5 times? shuffle? mean/scale/normalize?
#  2. dataloader = manager or location?

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
size = x_train.shape[1]
model = Autorec(size, hidden_dim_count).to(device)
optimizer = torch.optim.Rprop(model.parameters(), learning_rate)

# Print output
print('model', model)

# TODO :Questions
#  1. Bias?
#  2. model - identify for last layer / no reLu?
#  3. 'Rprop does not support sparse gradients'  https://pytorch.org/docs/master/_modules/torch/optim/rprop.html#Rprop

"""
4. Evaluation functions
"""
from s04_evaluation_functions import *

# Print output
print("Number of trainable parameters: {}".format(count_parameters(model)))

# TODO :Questions
#  1. custom loss does not have to inherit from module/class loss?
#  2. loss computed from sparse tensor > sparsity transmited to backprop/
#  only non-sparse indices undergo the gradient update?
#  3. weight/bias > both used for regularization?

"""
5. Running loss
"""
from s05_running_loss import *

# TODO : Questions:
#  1. sparsity in the training?

"""
6. Training Loop
"""

from s06_training_loop import *

tr_losses, te_losses = training_loop(model, optimizer, num_epochs, trainloader, testloader, device, regularization_term)

# TODO : Questions:
#  1. only update the weights with observed inputs / although we compute them for everything ?

"""
7. Investigate results
"""
from s07_plot_results import *

plot_results(tr_losses,te_losses)
#Check some reconstructions by the trained model
example_rating = sanity_check2(testloader)

#TODO: check this later

# Show images
preds = model(example_rating.to(device))
print(' '.join('%5s' % example_rating[j] for j in range(batch_size)))
# Print labels
print(' '.join('%5s' % preds[j] for j in range(batch_size)))

"""
7. Save results
"""

#TODO: export graphs/results

"""
8. Parameter hypertuning - with validation set
"""

# tune params with with validation set
# test deep AE