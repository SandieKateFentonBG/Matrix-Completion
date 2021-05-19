

def xremove_missing_ratings_1D(full_tensor, batch_data, placeholder = '99.'):

    indices = [] # TODO : check
    for data in batch_data:
        indices = (batch_data == float(placeholder)).nonzero(as_tuple=True)[0]
    for i in indices :
        full_tensor[i] = 0
    return full_tensor

def xremove_missing_ratings_2D(full_tensor, batch_data, placeholder = '99.'):

    missing_indices = [] # TODO : check
    for data in batch_data:
        missing_indices = (batch_data == float(placeholder)).nonzero(as_tuple=True)[0]
    for i in missing_indices:
        full_tensor[i] = torch.zeros_like(full_tensor[i])
    return full_tensor

def hypertune_model_parameters_regularization(model, reg_list):
    # TODO : regularization strength only influences the loss value > should not be hypertuned on a fix weight matrix but should be used to optimize weights
    [mystudy, x_train, x_test, x_val, trainloader, testloader, valloader, model, tr_losses, te_losses, val_losses,
     perc_acc] = model

    loss_list = []
    for val in reg_list:
        computed_loss = compute_autorec_te_loss(valloader, model, mystudy.device, val)
        loss_list.append(computed_loss)

    perc_acc = compute_perc_acc(valloader, mystudy.device, model)  # TODO : REMOVE : should be the same for all

    return reg_list, loss_list, perc_acc  # TODO : RMSE instead of loss; percentage acc instead of loss

    #1. optimizer.step() : performs a parameter update based on the current gradient
    # (stored in .grad attribute of a parameter) and the update rule.
    #2. Calling .backward() multiple times accumulates the gradient (by addition) for each parameter.
    # This is why you should call optimizer.zero_grad() after each .step() call.
    # Note that following the first .backward call, a second call is only possible
    # after you have performed another forward pass, unless you specify retain_graph=True