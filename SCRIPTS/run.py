
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
x_train, x_test, x_val = split_data(studied_data)    #TODO: repeat 5 times? shuffle? mean/scale/normalize?
trainloader, testloader, valloader = create_data_loader(x_train, x_test, x_val)
#TODO : at what stage should we do dataloader vs full data > batch size dim?

print("train :", x_train.shape)
print("test :", x_test.shape)
print("validate :", x_val.shape)

"""
2. Sanity check  
"""
from s02_sanity_check import *

check1 = sanity_check1(trainloader)#TODO:check later

"""
3. Model definition
"""
from s03_model_definition import * #TODO : if this is a class?

model = Autorec(input_dim[study], hidden_dim_count).to(device)
print(model)

"""
4. Evaluation functions
"""
from s04_evaluation_functions import *

print("Number of trainable parameters: {}".format(count_parameters(model))) #TODO: Number of trainable parameters: 7707199???
#loss_function = autorec_loss() #TODO: loss vs objective?
loss_function = nn.MSELoss() #TODO: remove

"""
5. Training Loop
"""
from s05_running_loss import *

tr_losses, te_losses = training_loop(model,learning_rate, num_epochs, trainloader, testloader)

"""
6. Investigate results
"""
from s07_plot_results import *

plot_results(tr_losses,te_losses)
#Check some reconstructions by the trained model #TODO: check this later
check2 = sanity_check2(testloader)

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