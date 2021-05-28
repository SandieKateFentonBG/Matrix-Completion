import torch
from mc_tr_03_data import *

import pickle

def pickleLoadMe(input_path, name, show = False):
    with open(input_path + name, 'rb') as handle:
        mydict = pickle.load(handle)
    if show:
        print(name)
        for k, v in mydict.items():
                print(' ', k, ' : ', v)
    return mydict

def model_print(model):
    # Print model's state_dict
    print("Model's state_dict:")
    for param_tensor in model.state_dict():
        print(param_tensor, "\t", model.state_dict()[param_tensor].size())

def optimizer_print(optimizer):
    # Print optimizer's state_dict
    print("Optimizer's state_dict:")
    for var_name in optimizer.state_dict():
        print(var_name, "\t", optimizer.state_dict()[var_name])


def save_model(model_result, PATH = None, inference = False):
    if not PATH :
        PATH = model_result.case_study.output_path + 'JesterDataset4.pth'
        print(PATH)
    if inference :
        torch.save(model_result.model.state_dict(), PATH)
    else:
        torch.save(model_result.model, PATH)

def load_model(PATH, inference = False):

    if inference:
        model = Autorec(*args, **kwargs)  # TODO : how use this? *args, **kwargs
        model.load_state_dict(torch.load(PATH))
        model.eval()  # TODO : do I need this? or is it not for custom models
    else:
        model = torch.load(PATH)
        model.train()  # TODO : do I need this?



def save_list_pth(result_list, PATH = None):
    if not PATH :
        PATH = result_list[0].case_study.output_path + 'JesterDataset4.pth'
        print(PATH)
    for i in range(len(result_list)):
        torch.save({ str(i)+'_model_state_dict': result_list[i].model.state_dict(),
            str(i) + '_optimizer_state_dict': result_list[i].optimizer.state_dict() }, PATH)

def load_list_pth (PATH, dim, train = True):

    model_list = []
    optim_list = []
    for i in range(dim):
        model_list.append(Autorec(*args, **kwargs))
        optim_list.append(torch.optim.Rprop(*args, **kwargs))

    checkpoint = torch.load(PATH)

    model_list[i].load_state_dict(checkpoint[str(i)+'_model_state_dict'])
    optim_list[i].load_state_dict(checkpoint[str(i) + '_optimizer_state_dict'])

    if train :
        model_list[i].train()
    else :
        model_list[i].eval()

    # https://pytorch.org/tutorials/beginner/saving_loading_models.html#saving-multiple-models-in-one-file


