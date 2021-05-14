def sanity_check1(dataloader):
    # Print some samples of dataset as a sanity check #TODO:check later
    # Get some random training ratings
    dataiter = iter(dataloader)
    example_ratings = dataiter.next() #

    print(example_ratings.shape)

def sanity_check2(testloader):

    # Get some random validation rating
    dataiter = iter(testloader)
    example_rating, _ = dataiter.next()

    print(example_rating.shape)