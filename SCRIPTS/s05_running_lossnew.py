from s04_evaluation_functions import *

def run_training(dataloader, optimizer, model, device, regul = 0.5): #todo : new name

    # Train model
    running_loss = 0.0
    running_acc = 0.0

    for batch_data in dataloader:  # the input image is also the label
        # Train model
        batch_data = batch_data.to(device) # b_d : 7700*1 ratings

        # Predict and get loss
        pred = model(batch_data).to(device) #
        #pred
        # x - forward prop > get first prediction - y
        loss = autorec_loss(pred, batch_data, model, regul)
        # f(x - y) - initial loss

        # Update model
        optimizer.zero_grad()
        # start with empty grad matrix (avoid accumulating gradient (by addition))
        loss.backward()
        # backpropagate loss from y > h > x in a chain rule - compute gradients
        optimizer.step()
        # parameter update based on the current gradient

        running_loss += loss.item()
        running_acc += compute_acc(pred, batch_data)

    tr_loss = running_loss / len(dataloader.dataset) #tr_loss = scalar
    tr_acc = 100 * running_acc/len(dataloader.dataset)

    dict = dict()
    dict['autorec_tr_loss'] = tr_loss
    dict['autorec_tr_acc'] = tr_acc

    return dict

    #print('>> TRAIN: Epoch {} completed | tr_loss: {:.4f} | tr_acc: {:.2f}%'.format(
    #    epoch_nr, running_loss / len(trainloader.dataset), tr_acc))


def compute_acc(pred, groundtruth, mask = 99.0, range = 20, round = False):

    bool_mask = groundtruth != mask
    if round:
        groundtruth = torch.round(groundtruth)
        pred = torch.round(pred)
    scaled_diff = torch.abs(groundtruth - pred) / range
    masked_perc = torch.sum(scaled_diff * bool_mask)/torch.sum(torch.ones_like(batch_data) * bool_mask)
    acc = 100 * (1 - masked_perc.item())

    return acc

def run_testing(testloader, model, device, regul = None):

    # Get validation results from testing set
    running_autorec_loss = 0
    running_rmse_loss = 0
    running_acc = 0
    #compute running accuracy
    with torch.no_grad():  # TODO : here no .backward() needed
        for batch_data in testloader:
            batch_data = batch_data.to(device)
            output = model(batch_data)
            running_autorec_loss += autorec_loss(output, batch_data, model, regul = regul) # TODO : regul on testing?
            running_rmse_loss += RMSELoss(output, batch_data)
            running_acc += compute_acc(output, batch_data)

    te_loss = running_autorec_loss / len(testloader.dataset)
    rmse_loss = running_rmse_loss / len(testloader.dataset)
    te_acc = 100 * running_acc/len(testloader.dataset)

    dict = dict()
    dict['autorec_te_loss'] = te_loss
    dict['rmse_te_loss'] = rmse_loss
    dict['te_acc'] = te_acc

    return dict

def running_loss_print(new_folder=False, folder = None, VISU = False, reference = None):

    if VISU :
        print('5. Running loss')
    if new_folder:
        from s10_helper_functions import mkdir_p
        mkdir_p(folder)
    if folder :
        with open(folder + reference + ".txt", 'a') as f:
            print('5. Running loss', file=f)
        f.close()







