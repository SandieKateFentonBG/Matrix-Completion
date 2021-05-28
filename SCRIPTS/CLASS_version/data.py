class Data:
    def __init__(self, xs, loaders):
        self.x_train = x[0]
        self.x_test = x[1]
        self.x_val = x[2]
        self.trainloader = loaders[0]
        self.testloader = loaders[1]
        self.valloader = loaders[2]

    def printMe(self, outputPath, reference, save=False, show=False):
        if save and not os.path.isdir(outputPath):
            os.makedirs(outputPath)
        toShow = [(self.x_train, 'training'), (self.x_test, 'test'), (self.x_val, 'validation')]
        for tens, name in toShow:
            if show:
                print(' ', name, ' : ', tens.shape)
                self.sanity_check()
            if save:
                print(' ', name, ' : ', tens.shape, file=open(self.output_path + title + ".txt", 'w+'))
        

    def sanity_check(self):
        print('SANITY CHECK : training, test, validation')
        for dataloader in [self.trainloader, self.testloader, self.valloader]:
            # Get random training ratings
            dataiter = iter(dataloader)
            example_ratings = dataiter.next()
            print(example_ratings)
