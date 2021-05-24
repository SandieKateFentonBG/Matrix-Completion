import torch

def autorec_loss(prediction, groundtruth, model, regul=None): 

    mask = groundtruth != 99.0
    sparse_loss_matrix = torch.square(mask *(groundtruth - prediction)) #TODO : check sparsity here
    loss = torch.sum(sparse_loss_matrix).to(groundtruth.device) #TODO : loss from sparse tensor > sparsity transmitted to backprop?

    if regul:
        V_frob_norm = sum([(p ** 2).sum() for p in model.hidden.parameters()]) # parameters = weights + bias
        W_frob_norm = sum([(q ** 2).sum() for q in model.predict.parameters()])
        loss += 0.5 * regul * (V_frob_norm + W_frob_norm)

    return loss


def RMSELoss(groundtruth, prediction):

    return torch.sqrt(torch.mean((groundtruth - prediction) ** 2))


def compute_acc(pred, groundtruth, mask=99.0, threshold=2):

    ones = torch.ones_like(groundtruth)
    sparse_mask = groundtruth != mask
    threshold_mask = torch.abs(groundtruth - pred) <= threshold

    return torch.sum(ones * threshold_mask * sparse_mask)/torch.sum(ones * sparse_mask)

def compute_acc2(pred, groundtruth, mask = 99.0, range = 20, round = False):

    bool_mask = groundtruth != mask
    if round:
        groundtruth = torch.round(groundtruth)
        pred = torch.round(pred)
    scaled_diff = torch.abs(groundtruth - pred) / range
    masked_perc = torch.sum(scaled_diff * bool_mask)/torch.sum(torch.ones_like(groundtruth) * bool_mask)
    acc = 100 * (1 - masked_perc.item())

    return acc



