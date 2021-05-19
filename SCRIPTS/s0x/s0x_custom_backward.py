


class CustomForwardBackward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
          with torch.enable_grad():
              output = ctx.g3.forward(input)
              ctx.save_for_backward(input, output)
          return ctx.f3.forward(input)
    @staticmethod
    def backward(ctx, grad_output):
          input, output = ctx.saved_tensors
          output.backward(grad_output, retain_graph=True)
          return input.grad




class CustomLayer_F(nn.Module):
    def __init__(self):
        super(CustomLayer_F, self).__init__()
        # defining all variables and operations

    def forward(self, input):
# computing the forward pass

class CustomLayer_G(nn.Module):
    def __init__(self):
        super(CustomLayer_G, self).__init__()
        # defining all variables and operations, different from F

    def forward(self, input):
# computing the forward pass, different from F