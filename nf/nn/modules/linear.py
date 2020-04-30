import nf
from .module import Module
from . import init


__all__ = ['Linear']

class Linear(Module):
    def __init__(self, in_features, out_features, bias=True):
        super(Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nf.zeros([out_features, in_features], requires_grad=True)
        self.bias = nf.zeros([out_features], requires_grad=True) if bias else None
        self.reset_parameters()

    def reset_parameters(self):
        # 5 ** 0.5 = 2.23606798
        init.kaiming_uniform_(self.weight, a=2.23606798)

        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / (fan_in ** 0.5)
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        if input.ndim == 2 and self.bias is not None:
            ret = input @ self.weight.T + self.bias
        else:
            output = input @ self.weight.T
            if self.bias is not None:
                output += self.bias
            ret = output
        return ret
