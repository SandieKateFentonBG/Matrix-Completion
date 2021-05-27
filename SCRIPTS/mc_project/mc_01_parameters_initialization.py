class case_study:

    def __init__(self, i_u_study = 0, database_group_split = 5, selected_group = 0, batch_size = 8, input_dim = [7699, 158],
                 hidden_dim = 500, num_epochs = 5, learning_rate = 0.01, regularization_term = 0.01, device = 'cuda:0',
                 date = '210525', project = 'Matrix-Completion', database = 'JesterDataset4/JesterDataset4.csv',
                 repository = "C:/Users/sfenton/Code/Repositories/" ):

        # Referencing
        self.date = date
        self.reference = 'group_' + str(selected_group) + '_h_dim_' + str(hidden_dim) + '_regul_' +str(regularization_term*1000)
        self.input_path = repository + str(project) + '/DATA/' + str(database)
        self.output_path = repository + project + '/RESULTS/' + date + '_results/'

        # Data split
        # focus of the study : { 0=item based, 1=user based }
        self.i_u_study = i_u_study
        # split database into data groups :
        self.database_group_split = database_group_split
        # select group to work on :
        self.selected_group = selected_group
        # iterate through data in batches :
        self.batch_size = batch_size

        # Neural network architecture
        # Dimension of input data :
        self.input_dim = input_dim # ratings_per_item, ratings_per_user
        # Dimension of hidden layer :
        self.hidden_dim = hidden_dim  # {10, 20, 40, 80, 100, 200, 300, 400, 500}

        # Training parameters
        # run trough data multiple times :
        self.num_epochs = num_epochs
        # Optimization learning rate :
        self.learning_rate = learning_rate  # {0.001, 0.01, 0.1, 1, 100, 1000}
        # Objective function regularization :
        self.regularization_term = regularization_term  # {0.05, 0.5, 5}
        # Device for computation
        self.device = device

        def __getitem__(self, key):
            return getattr(self, key)
            #items = mystudy.__dict__.items()
        def __setitem__(self, key, val):
            return setattr(self, key, val)


def studied_attribute(mystudy, attr, val):
    mystudy.__setitem__(attr, val)
    return attr, val

def case_study_print (mystudy, folder=None, new_folder=False, VISU = False):

    if new_folder:
        from SCRIPTS.CLEAN.s10_export_resultsnew import mkdir_p
        mkdir_p(mystudy.output_path)

    for k, v in mystudy.__dict__.items():
        if VISU:
            print(' ', k, ' : ', v)
        if folder:
            with open(mystudy.output_path + mystudy.reference + ".txt", 'a') as f:
                print(' ', k, ' : ', v, file=f)
            f.close()



