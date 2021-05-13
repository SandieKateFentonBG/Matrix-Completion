import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision

groundtruth = torch.Tensor([2, 1, 99.])
prediction = torch.Tensor([3, 2, 5])
print((4-2)**2)


loss = 0
for i in range(len(groundtruth)):
    if groundtruth[i] < 99.:
        print(i,groundtruth[i],prediction[i])
        loss += (groundtruth[i] - prediction[i])**2
        print(loss)

print('groundtruth', groundtruth)
print('prediction', prediction)
print (loss)

V,W = torch.Tensor([[3, 2, 5],[3, 2, 5]])
print(V)


def remove_missing_ratings_2D(full_tensor, batch_data, placeholder = '99.'):
    index = None
    for data in batch_data:
        missing_indices = (batch_data == float(placeholder)).nonzero(as_tuple=True)[0]
    for i in missing_indices:
        full_tensor[i] = torch.zeros_like(full_tensor[i])
    return full_tensor

batch_data = torch.Tensor([3, 2, 5])
full_tensor = torch.Tensor([[1, 2, 2, 7], [3, 1, 2, 4], [3, 1, 9, 4]])
end = remove_missing_ratings_2D(full_tensor, batch_data, placeholder='3')
print(end)


tensor([[0, 1],
        [0, 2],
        [1, 2]])