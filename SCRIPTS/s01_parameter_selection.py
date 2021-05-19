
#TODO: this should be a class...

class case_study:

    def __init__(self, study = 0, database_group_split = 5, selected_group = 0, batch_size = 8, input_dim = [7699, 158],
                 hidden_dim = 500, num_epochs = 5, learning_rate = 0.001, regularization_term = 0.5, device = 'cuda:0'):

        # Data split
        # focus of the study : { 0=item based, 1=user based }
        self.study = study
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

def input_study_display(mystudy, date, project, database, studied_attribute ='Default', studied_val = 'Default', folder=None,
                        new_folder=False, VISU = False):
    """
    1. File referencing
    """
    # Default
    reference = date + '_results_for_' + str(studied_attribute) + '_' + str(studied_val)
    input_path = "C:/Users/sfenton/Code/Repositories/" + str(project) + '/DATA/' + str(database)
    output_path = 'C:/Users/sfenton/Code/Repositories/' + project + '/RESULTS/' + date + '_results/'

    variables = mystudy.__dict__.keys()
    values = mystudy.__dict__.values()

    """
    3. Prints
    """
    if VISU :
        print('0. Paths')
        print(' reference : ', reference)
        print(' input_path : ', input_path)
        print(' output_path : ', output_path)
        print('1. Parameter selection')
        for k,v in mystudy.__dict__.items():
            print(' ', k, ' : ', v )

    """
    4. Exports
    """
    if new_folder:
        from s10_helper_functions import mkdir_p
        # Create new directory
        mkdir_p(output_path)
    if folder :
        with open(output_path + reference + ".txt", 'a') as f:
            print('0. Paths', file=f)
            print(' reference : ', reference, file=f)
            print(' input_path : ', input_path, file=f)
            print(' output_path : ', output_path, file=f)
            print('1. Parameter selection', file=f)
            for k, v in mystudy.__dict__.items():
                print(' ', k, ' : ', v, file=f)
        f.close()

    return reference, input_path, output_path



