#https://pytorch.org/docs/stable/_modules/torch/nn/modules/loss.html#MSELoss

"""
MSELoss(_Loss)
#https://pytorch.org/docs/master/generated/torch.Tensor.backward.html?highlight=backward#torch.Tensor.backward
    r: Creates a criterion that measures the mean squared error (squared L2 norm) between
    each element in the input :math:`x` and target :math:`y`.

Shape:
- Input::math: `(N, *)` where: math:`*` means, any number of additional dimensions
- Target::math: `(N, *)`, same shape as the input

criterion.backwards() should work

"""

Class autorec_loss(prediction, groundtruth, regul,theta ):
    #TODO : how do i add my rgularization in this / connect with gradients/BP
    # regul,theta : should these be in the inputs?
    # regul vs lr?
    """r_i = 'partially observed tensor'
    h(r_i, theta) = 'predicted rating'
    theta = [W, V] = 'weight matrix
    
    loss = (r-h)^2 + lamda*0.5*(W^2+V^2) 
    '"""
    return sum(prediction-groundtruth)^2 + 0.5*regul*(theta[0])^2+(theta[1])^2)


