from s04_evaluation_functions import *

def compute_tr_loss(dataloader, optimizer, model, device, regul = 0.5):

    # Train model
    running_loss = 0.0

    for batch_data in dataloader:  # the input image is also the label
        # Train model
        batch_data = batch_data.to(device) # b_d : 7700*1 ratings

        # Predict and get loss
        pred = model(batch_data).to(device)
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
        # #TODO: is this only done for the indices that participated in the loss computation?

        running_loss += loss.item()

    tr_loss = running_loss / len(dataloader.dataset)
    # #TODO: loss is a scalar - if I have to adapt it to choose which values to bp, I should keep loss as a vector?
    #   (dim 7700*1)?
    return tr_loss

    """# Sparsen model
    V, W = optimizer # TODO : compute all weights / only update for observed inputs ?
    Vsparse = remove_missing_ratings_2D(V, batch_data, placeholder='99.')
    Wsparse = remove_missing_ratings_2D(W, batch_data, placeholder='99.') #TODO : check dim of W = 500*7700 or 7700*500?
    sparse_optimzer = Vsparse, Wsparse
    optimizer = sparse_optimzer # TODO : check that this is a "deepcopy" > sparsity is saved into optimizer weights
    #here drop-out structure could be interesting as a reference - works only with certain elements
    #https://pytorch.org/docs/stable/generated/torch.nn.Dropout.html"""


def compute_te_loss(testloader, model, device, regul):

    # Get validation results from testing set
    running_loss = 0
    with torch.no_grad():  # TODO : here no .backward() needed
        for batch_data in testloader:
            batch_data = batch_data.to(device)
            output = model(batch_data)
            running_loss += autorec_loss(output, batch_data, model, regul) # TODO : regul on testing?

    te_loss = running_loss / len(testloader.dataset)

    return te_loss

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







