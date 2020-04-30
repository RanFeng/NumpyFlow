from nf.op import Operation
import numpy as np

__all__ = ['BatchNorm2d']






from time import time
class BatchNorm2d(Operation):
    path = None
    def __call__(self, a, weight, running_mean, running_var, training=False, eps=1e-05):
        self.variables = (a, weight, )
        self.running_mean = running_mean    # numpy
        self.running_var = running_var      # numpy
        self.training = training
        out = (a.data - running_mean)
        out *= weight.data
        out /= (running_var + eps) ** 0.5
        return out

    def backward(self, grad, **kwargs):
        a, b = self.variables
        delta_w = _conv2d_bpw(a.data, grad, self.stride, self.dilation)
        _w1, _w2, _w3, _w4 = b.shape
        delta_w = delta_w[:_w1, :_w2, :_w3, :_w4]
        self.variables[1].backward(delta_w)
