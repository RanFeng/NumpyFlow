import nf
from .module import Module
from . import init


__all__ = ['BatchNorm2d']



class BatchNorm2d(Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True):
        super(BatchNorm2d, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        if self.affine:
            self.weight = nf.ones((1,num_features,1,1), requires_grad=True)
            self.bias = nf.zeros((1,num_features,1,1), requires_grad=True)
        else:
            self.weight = None
            self.bias = None
        self.running_mean = nf.zeros((1,num_features,1,1))
        self.running_var = nf.ones((1,num_features,1,1))
        self.num_batches_tracked = nf.zeros(1)
        # self.reset_parameters()

    def reset_parameters(self):
        init.zeros_(self.running_mean)
        init.ones_(self.running_var)
        self.num_batches_tracked = nf.zeros(1)
        if self.affine:
            init.ones_(self.weight)
            init.zeros_(self.bias)

    def forward(self, a):
        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum
        if self.trainable:
            self.num_batches_tracked += 1
            if self.momentum is None:  # use cumulative moving average
                exponential_average_factor = 1.0 / float(self.num_batches_tracked)
            else:  # use exponential moving average
                exponential_average_factor = self.momentum
        running_mean = nf.mean(a, axis=(0,2,3), keepdims=True)
        running_var = nf.var(a, axis=(0,2,3), keepdims=True, ddof=1)
        out = (a - running_mean)
        if self.weight:
            out = out * self.weight
        out = out / (running_var + self.eps) ** 0.5
        if self.bias:
            out += self.bias
        self.running_var = (1 - exponential_average_factor) * self.running_var + exponential_average_factor * running_var
        self.running_mean = (1 - exponential_average_factor) * self.running_mean + exponential_average_factor * running_mean
        return out