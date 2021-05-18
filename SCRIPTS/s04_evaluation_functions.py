import torch

def count_parameters(model):

    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def autorec_loss(prediction, groundtruth, model, regul=None): #TODO : custom loss does not have to inherit from module/class loss?

    mask = groundtruth != 99.0
    full = torch.square(groundtruth - prediction)
    loss = torch.sum(full * mask).to(groundtruth.device) #TODO : loss from sparse tensor > sparsity transmitted to backprop?

    if regul:
        V_frob_norm = sum([(p ** 2).sum() for p in model.hidden.parameters()]) # parameters = weights + bias
        W_frob_norm = sum([(q ** 2).sum() for q in model.predict.parameters()]) #TODO: weight/bias > both used for regularization?
        loss += 0.5 * regul * (V_frob_norm + W_frob_norm)

    return loss

def evaluation_function_print(model, loss_function = 'custom autorec loss', new_folder=False, folder = None,
                              VISU = False, reference = None):

    if VISU :
        print('4. Evaluation function')
        print(" number of trainable parameters: {}".format(count_parameters(model)))
        print(' loss function : ', loss_function)
    if new_folder:
        from s10_helper_functions import mkdir_p
        mkdir_p(folder)
    if folder :
        with open(folder + reference + ".txt", 'a') as f:
            print('4. Evaluation function', file=f)
            print(" number of trainable parameters: {}".format(count_parameters(model)), file=f)
            print(' loss function : ', loss_function, file=f)
        f.close()

