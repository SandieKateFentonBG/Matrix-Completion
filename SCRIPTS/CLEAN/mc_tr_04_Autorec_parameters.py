class Autorec_parameters:

        def __init__(self,
                     hidden_dim=500, num_epochs=5, learning_rate=0.01, regularization_term=0.01, device='cuda:0'):


            # Neural network architecture
            # Dimension of hidden layer :
            self.hidden_dim = hidden_dim  # {10, 20, 40, 80, 100, 200, 300, 400, 500}
            # Optimization learning rate :
            self.learning_rate = learning_rate  # {0.001, 0.01, 0.1, 1, 100, 1000}
            # Objective function regularization :
            self.regularization_term = regularization_term  # {0.05, 0.5, 5}

            # Training parameters
            # run trough data multiple times :
            self.num_epochs = num_epochs
            # Device for computation
            self.device = device

            def __getitem__(self, key):
                return getattr(self, key)
                # items = mystudy.__dict__.items()

            def __setitem__(self, key, val):
                return setattr(self, key, val)

def mc_tr_params_print(myparams, mydata, reference, folder=None, new_folder=False, VISU=False):
    if new_folder:
        from mc_tr_05_Autorec_net import mkdir_p
        mkdir_p(mydata.output_path)

    for k, v in myparams.__dict__.items():
        if VISU:
            print(' ', k, ' : ', v)
        if folder:
            with open(folder + reference + ".txt", 'a') as f:
                print(' ', k, ' : ', v, file=f)
            f.close()
