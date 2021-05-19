
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

"""
1. Parameter initialization and data preparation
"""
# Load data
path = '/DATA/JesterDataset4/JesterDataset4.csv'
user_data = np.genfromtxt(path, delimiter=';', dtype='str', usecols = (i for i in range(1,159))) #first column removed

user_data = np.char.replace(user_data, ',', '.').astype(float)
#user_data = np.where(user_data > 20, 'NaN', user_data)

t = torch.Tensor([1, 2, 2])
print ((t == 2).nonzero(as_tuple=True)[0])

def remove_missing_ratings(full_tensor, batch_data, placeholder = '99.'):
    index = None
    for data in batch_data:
        indices = (batch_data == float(placeholder)).nonzero(as_tuple=True)[0]
    for i in indices :
        full_tensor[i] = 0
    return full_tensor


item_data = np.transpose(user_data)
combined_data = [item_data, user_data]
user_tensor = torch.tensor(user_data)
item_tensor = torch.tensor(item_data) #TODO: convert 99s?
print(user_data)
# split into training (81%), validation(9%) and test sets(10%)   #TODO: repeat 5 times? shuffle? mean/scale/normalize?
studied_data = combined_data[study]
N_split = int(0.9 * studied_data.shape[0])
x_temp = studied_data[:N_split, :]
x_test = studied_data[N_split:, :]
N_split2 = int(0.9 * x_temp.shape[0])
x_train = x_temp[:N_split2, :]
x_val = x_temp[N_split2:, :]

print("train :", x_train.shape, type(x_train))
print("test :", x_test.shape)
print("validate :", x_val.shape)

#construct dataset
trainset = torch.tensor(x_train)
testset = torch.tensor(x_test)
valset = torch.tensor(x_val)
print(len(trainset[0]))
# Create dataloaders #TODO: shuffle?

trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)
valloader = torch.utils.data.DataLoader(valset, batch_size=batch_size, shuffle=False)

sparse99 = next(iter(trainloader))
print('r', len(sparse99), type(sparse99))
full = torch.ones_like(sparse99)
print('full', full)
sparse0 = remove_missing_ratings(full, sparse99, placeholder=99.)

print('full', full)
print('sparse99', sparse99 )
print('sparse0', sparse0 )

"""
2. Sanity check  
"""
# Print some samples of dataset as a sanity check #TODO:check later

# Get some random training ratings
dataiter = iter(trainloader)
example_ratings = dataiter.next() #

print(example_ratings.shape)

"""
3. Model definition
"""

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

"""
4. Evaluation functions
"""

def compute_run_acc(logits, labels):
    # returns how often we were correct/how often we were wrong
    # > "boolean units" =/=loss, although are usually linked
    _, pred = torch.max(logits.data, 1)
    return (pred == labels).sum().item()

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def autorec_loss(prediction, groundtruth): #TODO: write this
    pass

"""
5. Training Loop
"""

# Instantiate components (model, optimizer, loss)

model = Autorec(input_dim[study], hidden_dim_count).to(device)
print(model)
print("Number of trainable parameters: {}".format(count_parameters(model))) #TODO: Number of trainable parameters: 7707199???
#loss_function = autorec_loss() #TODO: loss vs objective?
loss_function = nn.MSELoss() #TODO: remove
optimizer = optim.Rprop(model.parameters(), lr=learning_rate) #to compute weight matrix
best_val_loss = np.inf #TODO : check?

tr_losses = np.zeros(num_epochs)
te_losses = np.zeros(num_epochs)
val_losses = np.zeros(num_epochs)

for epoch_nr in range(num_epochs):

    print("Epoch {}:".format(epoch_nr))

    # Train model
    running_loss = 0.0
    for batch_data, _ in trainloader: # the input image is also the label
        # Train model

        # Put data on device
        batch_data = batch_data.to(device)

        # Predict and get loss
        output = model(batch_data)
        loss = loss_function(output, batch_data)

        # Update model #TODO: this should only be done in places where we have ratings
        optimizer.zero_grad() #fill weights with zeros at start

        loss.backward() #computes the gradients in a chain rule #TODO: but isn't this a scalar?
        # TODO : only update the weights with observed inputs / although we compute them for everything ?
        optimizer.step() # updates all the weight from the bp

        running_loss += loss.item()

        """
        6. Compute Results
        """

    # Print results
    tr_loss = running_loss / len(trainloader.dataset)
    print('>> TRAIN: Epoch {} completed | tr_loss: {:.4f}'.format(
        epoch_nr, running_loss / len(trainloader.dataset)))

    # Get validation results from testing set
    running_loss = 0
    # >> Your code goes here <<
    running_loss = 0
    with torch.no_grad():  # TODO : what is this
        for batch_data, _ in testloader:
            batch_data = batch_data.to(device)
            output = model(batch_data)
            running_loss += loss_function(output, batch_data).item()

    te_loss = running_loss / len(valloader.dataset)
    print('>> VALIDATION: Epoch {} | val_loss: {:.4f}'.format(epoch_nr, te_loss))

    tr_losses[epoch_nr] = tr_loss
    te_losses[epoch_nr] = te_loss

print('Training finished')

"""
7. Investigate results
"""

#Display results
plt.figure()
plt.plot(tr_losses, label='Training')
plt.plot(te_losses, label='Testing')
plt.title('Results')
plt.ylabel('Autorec_Loss')
plt.xlabel('Epochs')
plt.legend()
plt.show()

#Check some reconstructions by the trained model #TODO: check this later

# Get some random validation rating
dataiter = iter(testloader)
example_rating, _ = dataiter.next()

print(example_rating.shape)

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