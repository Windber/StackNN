'''
@author: lenovo
'''
import torch
a = 10
mingrad = 1.
maxgrad = 5.
 
class Sigmaid(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        output = 1 / ( 1 + torch.exp(-a * input))
        #output = input.clamp(min=0)
        ctx.save_for_backward(output)
        return output
      
    @staticmethod
    def backward(ctx, grad_output):
        output, = ctx.saved_tensors
        grad = a * output * (1 - output)
        return torch.clamp(grad, mingrad, maxgrad) * grad_output
