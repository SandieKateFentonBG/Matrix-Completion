import torch

def autorec_loss(prediction, groundtruth, model, regul=None): 

    mask = groundtruth != 99.0
    sparse_loss_matrix = torch.square(mask *(groundtruth - prediction)) #TODO : check sparsity here
    loss = torch.sum(sparse_loss_matrix)

    if regul:
        V_frob_norm = sum([(p ** 2).sum() for p in model.hidden.parameters()]) # parameters = weights + bias
        W_frob_norm = sum([(q ** 2).sum() for q in model.predict.parameters()])
        loss += 0.5 * regul * (V_frob_norm + W_frob_norm)

    return loss


def RMSELoss(groundtruth, prediction):
    return torch.sqrt(torch.mean((groundtruth - prediction) ** 2))


def result_accuracy(pred, groundtruth, mask=99.0, threshold=2): #TODO : use for inference as well?

    #a = pred.cpu().detach().numpy()
    ones = torch.ones_like(groundtruth)
    sparse_mask = groundtruth != mask
    threshold_mask = torch.abs(groundtruth - pred) <= threshold

    return torch.sum(ones * threshold_mask * sparse_mask)/torch.sum(ones * sparse_mask)




