import torch.nn as nn
import torch.nn.functional as F

class Autorec(nn.Module):

    def __init__(self, input_features, hidden_dim):
        super().__init__()
        #TODO : bias = True - see how to deal with this in s04 evaluation function - autorec loss
        #TODO :check the Nones
        #        
        self.hidden = nn.Linear(input_features, hidden_dim)
        self.predict = nn.Linear(hidden_dim, input_features)

        """
        (autorec_tr_acc = None, autorec_tr_loss = None, best_val_acc= None,
                 te_loss= None, te_acc= None, rmse_loss = None )
        self.rmse_losses = rmse_loss
        self.te_loss = te_loss
        self.te_acc = te_acc
        self.autorec_tr_loss = autorec_tr_loss
        self.autorec_tr_acc = autorec_tr_acc
        self.best_val_acc = best_val_acc
        """



    def forward(self, x):
        temp = self.hidden(x)
        x = F.relu(temp)
        x = self.predict(x) #TODO : identity for last layer / no reLu?
        return x

    def __setitem__(self, key, val):
        return setattr(self, key, val)

    def __getitem__(self, key):
        return getattr(self, key)
        # items = mystudy.__dict__.items()

def model_definition_print(model, optimizer, folder=False, new_folder=False, VISU = False, reference = None):

    if VISU :
        print('3. Model definition')
        print(' model', model)
        print(' optimizer', optimizer)
    if new_folder:
        from s10_helper_functions import mkdir_p
        mkdir_p(folder)
    if folder :
        with open(folder + reference + ".txt", 'a') as f:
            print('3. Model definition', file=f)
            print(' model', model, file=f)
            print(' optimizer', optimizer, file=f) #TODO : 'Rprop does not support sparse gradients'?
        f.close()
