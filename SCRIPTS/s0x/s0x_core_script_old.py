"""
1. File referencing and parameter selection
"""
from s01_parameter_selection import *

#Reference
project = 'Matrix-Completion'
database = 'JesterDataset4/JesterDataset4.csv'
date = "210518"
#test_count = str(0)

#Parameter optionss

selected_group_parameters = [0, 1, 2, 3, 4] # TODO : do it on all then keep only best results?
hidden_dim_parameters = [10, 20, 40, 80, 100, 200, 300, 400, 500]
num_epochs_parameters = [5, 10, 15] # TODO : choice?
learning_rate_parameters = [0.001, 0.01, 0.1, 1, 100, 1000] # TODO : choice?
regularization_term_parameters = [0.001, 0.01, 0.1, 1, 100, 1000]

#Initialize study
mystudy = case_study(selected_group=selected_group_parameters[0], hidden_dim = hidden_dim_parameters[8],
                     num_epochs = num_epochs_parameters[0], learning_rate = learning_rate_parameters[1],
                     regularization_term = regularization_term_parameters[0])
attr = 'selected_group' #'hidden_dim', 'num_epochs', 'learning_rate', 'regularization_term'
val = mystudy.__getattribute__(attr)
reference, input_path, output_path = input_study_display(mystudy, date, attr, val,
                                                         project, database, folder=True, new_folder=True)

"""
2. Parameter initialization and data preparation
"""
from s02_data_initialization import *

# format data
user_data = load_data(input_path)
sections = split_data(mystudy.database_group_split, user_data)
test_section = sections[mystudy.selected_group]
studied_data = selected_data(test_section, mystudy.i_u_study)
x_train, x_test, x_val = split_train_test_validate_data(studied_data)
trainloader, testloader, valloader = create_data_loader(x_train, x_test, x_val, mystudy.batch_size)
data_initialization_print(x_train, x_test, x_val, folder = output_path)
#check1 = sanity_check(trainloader)

# TODO :Questions
#  1. repeat 5 times - for each group? shuffle? mean/scale/normalize?

"""
3. Model definition
"""
from s03_model_definition import *

# Instantiate model and optimizer
size = x_train.shape[1]
model = Autorec(size, mystudy.hidden_dim).to(mystudy.device)
optimizer = torch.optim.Rprop(model.parameters(), mystudy.learning_rate)
model_definition_print(model, optimizer, folder=output_path)

# TODO :Questions
#  1. Bias?
#  2. model - identify for last layer / no reLu?
#  3. 'Rprop does not support sparse gradients'  https://pytorch.org/docs/master/_modules/torch/optim/rprop.html#Rprop

"""
4. Evaluation functions
"""
from s04_evaluation_functions import *

evaluation_function_print(model, loss_function = 'custom autorec loss', folder = output_path)

# TODO :Questions
#  1. custom loss does not have to inherit from module/class loss?
#  2. loss computed from sparse tensor > sparsity transmitted to backprop/
#  only non-sparse indices undergo the gradient update?
#  3. weight/bias > both used for regularization?

"""
5. Running loss
"""
from s05_running_loss import *

running_loss_print(folder = output_path)

# TODO : Questions:
#  1. sparsity in the training?
#  2. only update the weights with observed inputs
#  / although we compute them for everything ?
#  3. Regul on testing?

"""
6. Training Loop
"""

from s06_training_loop import *

tr_losses, te_losses, val_losses = training_loop(model, optimizer, mystudy.num_epochs, trainloader, testloader, mystudy.device,
                                     mystudy.regularization_term, folder = output_path)

"""
7. Investigate results
"""
from s07_investigate_results import *

plot_results(tr_losses,te_losses, reference, output_path)
example_rating = sanity_check(testloader)
perc_acc = compute_perc_acc(testloader, mystudy.device, model, round = False, folder = output_path)
#preds = model(example_rating.to(mystudy.device)) #sanity check

#TODO :Questions:
# 1. percentage of accuracy - how representative is pred-truth/20?
# 2. use this to hypertune my parameters ? how? validation set?

"""
8. Parameter hypertuning - with validation set
"""
"""
•	95% conﬁdence intervals on RMSE were ±0.003 or less in each experiment
•	regularisation strength λ ∈ {0.001, 0.01, 0.1, 1, 100, 1000} 
•	latent dimension k ∈ {10, 20, 40, 80, 100, 200, 300, 400, 500} 

plot 1 : RMSE of I-AutoRec on dataset 1 as the number of hidden units k varies.
plot 2 : RMSE of I-AutoRec on dataset 1 as the regularisation strength varies.
plot 1 : RMSE of I-AutoRec on dataset 2 as the number of hidden units k varies.
plot 2 : RMSE of I-AutoRec on dataset 2 as the regularisation strength varies.
> Do it for all 5 groups > take best results?

"""

#TODO :tune params with with validation set
#TODO : test deep AE
#TODO : test other dataset !!

"""
9. Save results
"""
#TODO: create plots/studied attribute
#TODO: export graphs/results
#TODO: export results as CSV

