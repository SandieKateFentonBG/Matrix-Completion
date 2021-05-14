import matplotlib.pyplot as plt

def plot_results(tr_losses,te_losses):

    #Display results
    plt.figure()
    plt.plot(tr_losses, label='Training')
    plt.plot(te_losses, label='Testing')
    plt.title('Results')
    plt.ylabel('Autorec_Loss')
    plt.xlabel('Epochs')
    plt.legend()
    plt.show()