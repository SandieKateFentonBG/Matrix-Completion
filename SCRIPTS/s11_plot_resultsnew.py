import matplotlib.pyplot as plt
import torch

def plot_results(tr_losses,te_losses, reference = None, folder = None, VISU = False):

    #Display results
    plt.figure()
    plt.plot(tr_losses, label='Training')
    plt.plot(te_losses, label='Testing')
    plt.title(reference)
    plt.ylabel('Autorec_Loss')
    plt.xlabel('Epochs')
    plt.legend()
    if folder :
        plt.savefig(folder + reference + '_Epochs' + '-' + 'Autorec_Loss' + '.png')
    if VISU:
        plt.show()

def plot_x_y_graph(x_list, y_list, x_label, y_label, title=None, folder=None, VISU=False):

    import matplotlib.pyplot as plt

    plt.figure()
    plt.plot(xdata = x_list, ydata = y_list,  label='Validation')
    if not title:
        title = x_label + 'as a function of' + y_label
    plt.title(reference)
    plt.ylabel(y_label)
    plt.xlabel(x_label)
    plt.legend()
    if folder:
        plt.savefig(folder + title + '.png')
    if VISU:
        plt.show()


def plot_sns_graph(x_list, y_list, x_label, y_label, title=None, figure_size=(12,15), folder=None, plot=False):
    import seaborn as sns
    import matplotlib.pyplot as plt
    import pandas as pd
    df = pd.DataFrame(list(zip(x_list, y_list)), columns =[x_label, y_label])
    fig, ax = plt.subplots(figsize=figure_size)
    if not title :
        title = x_label + 'as a function of' + y_label
    ax.set_title(title)
    sns.scatterplot(data=df, x=x_label, y=y_label, hue=y_label)
    if folder :
        plt.savefig(folder + '/'+ x_label + '-' + y_label +'.png')
    if plot:
        plt.show()
