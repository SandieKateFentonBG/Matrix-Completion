
#TODO: this should be a class...

def input_study_display(date, test_count, project, database, folder=False, new_folder=False):
    """
    1. File referencing
    """
    # Default
    reference = date + '_results_' + test_count
    input_path = "C:/Users/sfenton/Code/Repositories/" + project + '/DATA/' + database
    output_path = 'C:/Users/sfenton/Code/Repositories/' + project + '/RESULTS/' + date + '_results/'

    """
    2. Parameter selection
    """
    # Data split
    # focus of the study : { 0=item based, 1=user based }
    study = 0
    # split database into data groups :
    database_group_split = 5
    # select group to work on :
    selected_group = 0
    # iterate through data in batches :
    batch_size = 8

    # Neural network architecture
    # Dimension of input data :
    input_dim = [7699, 158]  # ratings_per_item, ratings_per_user
    # Dimension of hidden layer :
    hidden_dim = 500  # {10, 20, 40, 80, 100, 200, 300, 400, 500}

    # Training parameters
    # run trough data multiple times :
    num_epochs = 5
    # Optimization learning rate :
    learning_rate = 0.001  # {0.001, 0.01, 0.1, 1, 100, 1000}
    # Objective function regularization :
    regularization_term = 0.5  # {0.001, 0.01, 0.1, 1, 100, 1000
    # Device for computation
    device = 'cuda:0'

    attributes = ['reference', 'input_path', 'output_path', 'focus of the study', 'split database into data groups',  ]
    values = []

    """
    3. Prints
    """
    print('0. Paths')
    print('reference : ', date +'_results_' + test_count)
    print('input_path : ', "C:/Users/sfenton/Code/Repositories/" + project + '/' + database)
    print('output_path : ', 'C:/Users/sfenton/Code/Repositories/' + project + '/RESULTS/' + date + '_results/')
    print('1. Parameter selection')
    print('# Data split')
    print(' focus of the study : ', study, '{ 0=item based, 1=user based }')
    print(' split database into data groups :', database_group_split)
    print(' select group to work on :', selected_group)
    print(' iterate through data in batches :', batch_size)
    print('# Neural network architecture')
    print(' Dimension of input data :', input_dim)
    print(' Dimension of hidden layer :', hidden_dim)
    print('# Training parameters')
    print(' Number of epochs (run trough data multiple times) :', num_epochs)
    print(' Optimization learning rate :', learning_rate)
    print(' Objective function regularization :', regularization_term)
    print(' Device for computation : ', device)

    """
    4. Exports
    """

    if new_folder:
        from s09_helper_functions import mkdir_p
        # Create new directory
        mkdir_p(output_path)
    if folder :
        with open(output_path + "results.txt", 'a') as f:

            print('0. Paths', file=f)
            print('reference : ', date +'_results_' + test_count, file=f)
            print('input_path : ', "C:/Users/sfenton/Code/Repositories/" + project + '/' + database, file=f)
            print('output_path : ', 'C:/Users/sfenton/Code/Repositories/' + project + '/RESULTS/' + date + '_results/', file=f)
            print('1. Parameter selection', file=f)
            print('# Data split', file=f)
            print(' focus of the study : ', study, '{ 0=item based, 1=user based }', file=f)
            print(' split database into data groups :', database_group_split, file=f)
            print(' select group to work on :', selected_group, file=f)
            print(' iterate through data in batches :', batch_size, file=f)
            print('# Neural network architecture', file=f)
            print(' Dimension of input data :', input_dim, file=f)
            print(' Dimension of hidden layer :', hidden_dim, file=f)
            print('# Training parameters', file=f)
            print(' Number of epochs (run trough data multiple times) :', num_epochs, file=f)
            print(' Optimization learning rate :', learning_rate, file=f)
            print(' Objective function regularization :', regularization_term, file=f)
            print(' Device for computation : ', device, file=f)

        f.close()



    """
    5. Returns
    """

    return reference, input_path, output_path, study, database_group_split, selected_group, batch_size, input_dim, hidden_dim, num_epochs, learning_rate, regularization_term, device



