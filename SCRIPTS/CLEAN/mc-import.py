# import pickle
#
# input_path = 'C:/Users/sfenton/Code/Repositories/Matrix-Completion/RESULTS/210528_results/'
#
# def picklePrintMe(input_path, name):
#     with open(input_path + name, 'rb') as handle:
#         mydict = pickle.load(handle)
#     print(name)
#     for k, v in mydict.items():
#             print(' ', k, ' : ', v)
#
#
# picklePrintMe(input_path,'calibration_dict.pickle')
# picklePrintMe(input_path,'training_dict.pickle')

import torch
import torch.nn as nn
a = torch.randn(4, 4)
print(a)
b = torch.mean(a)

c= torch.sum(a)

print(b)
print(c)



loss = nn.MSELoss()
input = torch.randn(3, 5, requires_grad=True)
target = torch.randn(3, 5)
output = loss(input, target)

print(input)
print(target)
print(output)