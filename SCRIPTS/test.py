import torch

criterion = torch.nn.MSELoss()
a = torch.Tensor([[1, 2, 3], [1, 2, 3]]).view(-1, 2)
b = torch.Tensor([[1, 2, 2], [1, 2, 2]]).view(-1, 2)
c = a*b
print(a, a.shape)
print(b, b.shape)
print(c, c.shape)

def printer():
    for i in range(len(groundtruth)):
        print(i)
        print('gt', type(groundtruth[i]), len(groundtruth[i]), groundtruth[i])
        print('pred', type(prediction[i]), len(prediction[i]), prediction[i])
        print('diff', type(groundtruth[i] - prediction[i]), len(groundtruth[i] - prediction[i]),
              groundtruth[i] - prediction[i])
        print('mask', type(mask[i]), len(mask[i]), mask[i])
        print('square', type(full[i]), len(full[i]), full[i])
        print('mult', type((full * mask)[i]), len((full * mask)[i]), (full * mask)[i])
        start = 0
        count = 0
        for j, k in zip(mask[i], full[i]):
            if j:
                count += 1
                start += k
                print(count, k, start)

        print('loss', loss)



a = torch.Tensor([True, False, False, False, False, False])
b = torch.Tensor([1, 2, 3, 4.5, 2.3, 1])
c = a*b
print(a, a.shape)
print(b, b.shape)
print(c, c.shape)