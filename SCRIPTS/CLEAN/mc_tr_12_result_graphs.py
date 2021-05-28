import matplotlib.pyplot as plt
import torch

def plot_results(loss_a, loss_b, label_a, label_b, reference = None, folder = None, VISU = False):

    #Display results
    plt.figure()
    plt.plot(loss_a, label=label_a)
    plt.plot(loss_b, label=label_b)
    plt.title(reference)
    plt.ylabel('Autorec_Loss')
    plt.xlabel('Epochs')
    plt.legend()
    if folder :
        plt.savefig(folder + reference + '_Epochs' + '-' + 'Autorec_Loss' + '.png')
    if VISU:
        plt.show()

def plot_all_results(model_results, y_label = 'AE Loss', x_label = 'Epochs', reference = None, folder = None, VISU = False):

    plt.figure()
    plt.plot(model_results.te_losses, label='autorec_te_losses')
    plt.plot(model_results.rmse_te_losses, label='rmse_te_losses')
    plt.plot(model_results.te_accuracies, label='te_accuracies')
    plt.plot(model_results.tr_losses, label='autorec_tr_losses')
    plt.plot(model_results.tr_accuracies, label='tr_accuracies')

    plt.title(reference)
    plt.ylabel(y_label)
    plt.xlabel(x_label)
    plt.legend()
    if folder :
        plt.savefig(folder + reference + y_label + '-' + x_label +'.png')
    if VISU:
        plt.show()



def plot_sns_graph(x_list, y_list, x_label, y_label, title=None, figure_size=(12,15), folder=None, plot=False):
    import seaborn as sns
    import matplotlib.pyplot as plt
    import pandas as pd
    df = pd.DataFrame(list(zip(x_list, y_list)), columns =[x_label, y_label])
    fig, ax = plt.subplots(figsize=figure_size)
    if not title :
        title = y_label + ' as a function of ' + x_label
    ax.set_title(title)
    sns.scatterplot(data=df, x=x_label, y=y_label, hue=y_label)
    if folder :
        plt.savefig(folder + '/'+ x_label + '-' + y_label +'.png')
    if plot:
        plt.show()

