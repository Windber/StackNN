'''
@author: lenovo
'''
import torch
a = 20
mingrad = 1.
class Sigmaid(torch.torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        print("in sigmaid forward")
        output = 1 / ( 1 + torch.exp(-a * input))
        ctx.save_for_backward(output)
        print("out sigmaid forward")
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        print("in sigmaid backward")
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input < 0] = 0
        print("out sigmaid backward")
        return grad_input
#         output, = ctx.saved_tensors
#         grad_input = grad_output.clone()
#         return gr ad_input