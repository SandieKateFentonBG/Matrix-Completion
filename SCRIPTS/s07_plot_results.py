import matplotlib.pyplot as plt

def plot_results(tr_losses,te_losses, reference = None, folder = None):

    #Display results
    plt.figure()
    plt.plot(tr_losses, label='Training')
    plt.plot(te_losses, label='Testing')
    plt.title(reference)
    plt.ylabel('Autorec_Loss')
    plt.xlabel('Epochs')
    plt.legend()
    if folder :
        plt.savefig(folder + 'Epochs' + '-' + 'Autorec_Loss' + '.png')
    plt.show()