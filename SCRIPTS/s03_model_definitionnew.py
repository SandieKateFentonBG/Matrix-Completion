import torch.nn as nn
import torch.nn.functional as F

class Autorec(nn.Module):

    def __init__(self, input_features, hidden_dim):
        super().__init__()

        self.hidden = nn.Linear(input_features, hidden_dim)
        self.predict = nn.Linear(hidden_dim, input_features)

    def forward(self, x):
        temp = self.hidden(x)
        x = F.relu(temp)
        x = self.predict(x) #identity for last layer - allows negative predicted results
        return x

def count_parameters(model):

    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def model_definition_print(model, optimizer, folder=False, new_folder=False, VISU = False, reference = None):

    if new_folder:
        from s10_export_resultsnew import mkdir_p
        mkdir_p(folder)
    if VISU :
        print(' model', model)
        print(' optimizer', optimizer)
        print(" number of trainable parameters: {}".format(count_parameters(model)))

    if folder :
        with open(folder + reference + ".txt", 'a') as f:
            print(' model', model, file=f)
            print(' optimizer', optimizer, file=f)
            print(" number of trainable parameters: {}".format(count_parameters(model)))
        f.close()
