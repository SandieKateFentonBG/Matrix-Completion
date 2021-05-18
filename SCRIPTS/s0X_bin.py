

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



    #1. optimizer.step() : performs a parameter update based on the current gradient
    # (stored in .grad attribute of a parameter) and the update rule.
    #2. Calling .backward() multiple times accumulates the gradient (by addition) for each parameter.
    # This is why you should call optimizer.zero_grad() after each .step() call.
    # Note that following the first .backward call, a second call is only possible
    # after you have performed another forward pass, unless you specify retain_graph=True