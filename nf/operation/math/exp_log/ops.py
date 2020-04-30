from nf.op import Operation
import numpy as np


__all__ = [
    "Exp",
    "Log",
    "Log2",
    "Log10",
]


class Exp(Operation):
    def __call__(self, a):
        self.variables = (a,)
        return np.exp(a.data)

    def backward(self, grad, **kwargs):
        a = self.variables[0]
        self.variables[0].backward(grad * np.exp(a.data))

class Log(Operation):
    def __call__(self, a):
        self.variables = (a,)
        return np.log(a.data)

    def backward(self, grad, **kwargs):
        a = self.variables[0]
        self.variables[0].backward(grad / a.data)

class Log2(Operation):
    def __call__(self, a):
        self.variables = (a,)
        return np.log2(a.data)

    def backward(self, grad, **kwargs):
        a = self.variables[0]
        self.variables[0].backward(grad / a.data * np.log(2))


class Log10(Operation):
    def __call__(self, a):
        self.variables = (a,)
        return np.log10(a.data)

    def backward(self, grad, **kwargs):
        a = self.variables[0]
        self.variables[0].backward(grad / a.data * np.log(10))

