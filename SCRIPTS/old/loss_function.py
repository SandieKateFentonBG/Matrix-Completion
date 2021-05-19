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

""""
def mse_loss(
    input: Tensor,
    target: Tensor,
    size_average: Optional[bool] = None,
    reduce: Optional[bool] = None,
    reduction: str = "mean",
) -> Tensor:
    rémse_loss(input, target, size_average=None, reduce=None, reduction='mean') -> Tensor

    Measures the element-wise mean squared error.

    See :class:`~torch.nn.MSELoss` for details.

    if has_torch_function_variadic(input, target):
        return handle_torch_function(
            mse_loss, (input, target), input, target, size_average=size_average, reduce=reduce, reduction=reduction
        )
    if not (target.size() == input.size()):
        warnings.warn(
            "Using a target size ({}) that is different to the input size ({}). "
            "This will likely lead to incorrect results due to broadcasting. "
            "Please ensure they have the same size.".format(target.size(), input.size()),
            stacklevel=2,
        )
    if size_average is not None or reduce is not None:
        reduction = _Reduction.legacy_get_string(size_average, reduce)

    expanded_input, expanded_target = torch.broadcast_tensors(input, target)
    return torch._C._nn.mse_loss(expanded_input, expanded_target, _Reduction.get_enum(reduction))
"""""