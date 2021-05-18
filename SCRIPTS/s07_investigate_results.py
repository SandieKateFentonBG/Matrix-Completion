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

def compute_perc_acc(testloader, device, model, round = True, folder = None, VISU = False, reference = None):

    summed_perc = 0
    summed_ratings = 0
    with torch.no_grad():  # no .backward() needed
        for batch_data in testloader:
            batch_data = batch_data.to(device) #TODO : I can be sure this is done with latest parameter values? without changing them?
            bool_mask = batch_data != 99.0
            output = model(batch_data)

            if round:
                batch_data = torch.round(batch_data)
                output = torch.round(output)
            perc = torch.abs(batch_data - output) / (torch.ones_like(batch_data)*20)  #TODO :how representative is this?
            masked_perc = perc * bool_mask
            rating_count = torch.ones_like(batch_data) * bool_mask
            summed_perc += torch.sum(masked_perc)
            summed_ratings += torch.sum(rating_count)
    acc = 100 - ((summed_perc / summed_ratings).item() * 100) #TODO :how can I round this?
    if VISU :
        print('7. Investigate results')
        print('prediction accuracy :', acc, '% ')
        print('(', summed_perc.item(), ' total margin (pred-truth/20) for ',
              summed_ratings.item(), ' predicted ratings )')
    if folder :
        with open(folder + reference + ".txt", 'a') as f:
            print('7. Investigate results', file=f)
            print('prediction accuracy :', acc, '%', file=f)
            print('(', summed_perc.item(), ' total margin (pred-truth/20) for ',
                  summed_ratings.item(), ' predicted ratings )', file=f)
    return acc