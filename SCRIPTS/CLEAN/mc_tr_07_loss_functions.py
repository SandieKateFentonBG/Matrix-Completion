from mc_tr_06_evaluation_criteria import *

def training_loss(dataloader, optimizer, model, myparams, threshold): #todo : new name

    # Train model
    running_loss = 0.0
    running_acc = 0.0

    for batch_data in dataloader:  # the input image is also the label
        # Train model
        batch_data = batch_data.to(myparams.device) # b_d : 7700*1 ratings

        # Predict and get loss
        pred = model(batch_data).to(myparams.device)
        # x > forward prop > get first prediction - y
        loss = autorec_loss(pred, batch_data, model, regul = myparams.regularization_term)
        #loss, sparse_loss_matrix = autorec_loss2(pred, batch_data, model, regul = myparams.regularization_term)

        # f(x - y) - initial loss

        # Update model
        optimizer.zero_grad()
        # start with empty grad matrix (avoid accumulating gradient (by addition))
        loss.backward()
        #sparse_loss_matrix.backward()#todo
        # backpropagate loss from y > h > x in a chain rule - compute gradients
        optimizer.step()
        # parameter update based on the current gradient

        running_loss += loss.item()
        running_acc += result_accuracy(pred, batch_data, threshold = threshold)

    tr_loss = running_loss / len(dataloader.dataset) #tr_loss = scalar
    tr_acc = running_acc/len(dataloader.dataset)

    loss_dict = dict()
    loss_dict['autorec_tr_loss'] = tr_loss
    loss_dict['autorec_tr_acc'] = tr_acc

    return loss_dict




def testing_loss(testloader, model, myparams, threshold ):

    # Get validation results from testing set
    running_autorec_loss = 0
    running_rmse_loss = 0
    running_acc = 0
    with torch.no_grad():  # weights kept constant - no .backward() needed
        for batch_data in testloader:
            batch_data = batch_data.to(myparams.device)
            output = model(batch_data)
            running_autorec_loss += autorec_loss(output, batch_data, model, regul = None) # no regul when testing
            running_rmse_loss += RMSELoss(batch_data, output)
            running_acc += result_accuracy(output, batch_data, threshold = threshold)

    te_loss = running_autorec_loss / len(testloader.dataset)
    rmse_loss = running_rmse_loss / len(testloader.dataset)
    te_acc = running_acc/len(testloader.dataset)

    loss_dict = dict()
    loss_dict['autorec_te_loss'] = te_loss
    loss_dict['rmse_te_loss'] = rmse_loss
    loss_dict['te_acc'] = te_acc

    return loss_dict







