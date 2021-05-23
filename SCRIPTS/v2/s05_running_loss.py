from SCRIPTS.v2.s04_evaluation_functions import *

def autorec_tr_loss(dataloader, optimizer, model, device, regul = 0.5):

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
        running_acc += compute_run_acc(logits, batch_labels)


    tr_loss = running_loss / len(dataloader.dataset) #tr_loss = scalar
    tr_acc = 100 * running_acc/len(dataloader.dataset)

    return tr_loss, tr_acc

    print('>> TRAIN: Epoch {} completed | tr_loss: {:.4f} | tr_acc: {:.2f}%'.format(
        epoch_nr, running_loss / len(trainloader.dataset), tr_acc))

    # Get validation results
    running_acc = 0
    with torch.no_grad():
        for batch_data, batch_labels in valloader:
            batch_data, batch_labels = batch_data.to(device), batch_labels.to(device)
            logits = model(batch_data)
            running_acc += compute_run_acc(logits, batch_labels)

def compute_run_acc(logits, labels):
    _, pred = torch.max(logits.data, 1)
    return (pred == labels).sum().item()

def compute_te_loss(testloader, model, device, regul = None):

    # Get validation results from testing set
    running_autorec_loss = 0
    running_rmse_loss = 0
    #compute running accuracy
    with torch.no_grad():  # TODO : here no .backward() needed
        for batch_data in testloader:
            batch_data = batch_data.to(device)
            output = model(batch_data)
            running_autorec_loss += autorec_loss(output, batch_data, model, regul = regul) # TODO : regul on testing?
            running_rmse_loss += RMSELoss(output, batch_data)
    te_loss = running_autorec_loss / len(testloader.dataset)
    rmse_loss = running_rmse_loss / len(testloader.dataset)

    return te_loss, rmse_loss

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







