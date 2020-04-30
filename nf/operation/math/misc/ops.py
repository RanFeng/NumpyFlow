from nf.op import Operation
import numpy as np
__all__ = ["Abs", "Cbrt", "Maximum", "Minimum"]


class Abs(Operation):
    def __call__(self, a):
        self.variables = (a,)
        return np.abs(a.data)

    def backward(self, grad, **kwargs):
        a = self.variables[0]
        grad = grad * np.piecewise(
            a.data, [a.data < 0, a.data == 0, a.data > 0], [-1, np.nan, 1]
        )
        self.variables[0].backward(grad)

class Cbrt(Operation):
    def __call__(self, a):
        self.variables = (a,)
        return np.cbrt(a.data)

    def backward(self, grad, **kwargs):
        a = self.variables[0]
        self.variables[0].backward(grad / (3 * np.cbrt(a.data ** 2)))

class Maximum(Operation):
    def __call__(self, a, b):
        self.variables = (a, b)
        self.greater_than_mask = a.data > b.data
        self.equal_mask = a.data == b.data
        return np.where(self.greater_than_mask, a.data, b.data)

    def backward(self, grad, **kwargs):
        self.variables[0].backward(grad * self.greater_than_mask)
        mask = np.logical_not(self.greater_than_mask)
        if mask.ndim:
            np.logical_not(mask, out=mask, where=self.equal_mask)
        elif self.equal_mask:
            mask = np.logical_not(mask)
        self.variables[1].backward(grad * mask)


class Minimum(Operation):
    def __call__(self, a, b):
        self.variables = (a, b)
        self.less_than_mask = a.data < b.data
        self.equal_mask = a.data == b.data
        return np.where(self.less_than_mask, a.data, b.data)

    def backward(self, grad, **kwargs):
        self.variables[0].backward(grad * self.less_than_mask)
        mask = np.logical_not(self.less_than_mask)
        if mask.ndim:
            np.logical_not(mask, out=mask, where=self.equal_mask)
        elif self.equal_mask:
            mask = np.logical_not(mask)
        self.variables[1].backward(grad * mask)