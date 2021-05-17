"""def sanity_check(dataloader, test_print = False, new_folder=False, folder = None):
    # Get random training ratings
    dataiter = iter(dataloader)
    example_ratings = dataiter.next()
    if test_print:
        print(example_ratings.shape)
    if new_folder:
        from s09_helper_functions import mkdir_p
        mkdir_p(folder)
    if folder :
        with open(folder + "results.txt", 'a') as f:
            print('2. Sanity check', file=f)
            print('example_ratings : ', example_ratings, file=f)
        f.close()
    return example_ratings"""

"""def sanity_check2(testloader, test_print = False, folder=False, new_folder=False, output_path = None):

    # TODO
    dataiter = iter(testloader)
    example_rating = dataiter.next()
    
    if test_print:
        print(example_rating.shape)
    if new_folder:
        from s09_helper_functions import mkdir_p
        mkdir_p(output_path)
    if folder :
        with open(output_path + "results.txt", 'a') as f:
            print('2. Sanity check', file=f)
            print('example_rating : ', example_rating, file=f)
        f.close()
    
    print(example_rating.shape)
    return example_rating"""

