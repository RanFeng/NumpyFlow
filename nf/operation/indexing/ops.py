from nf.op import Operation
import numpy as np


__all__ = ['GetItem','Where']

class GetItem(Operation):

    def __call__(self, a, index):
        """
        使得Tensor能像numpy数组一样被index访问并支持反向传播，例如a[3],a[3,2,1]等等
        """
        self.variables = (a,)
        self.index = index
        return a.data[index]

    def backward(self, grad, **kwargs):
        a = self.variables[0]
        out = np.zeros_like(a.data)
        grad = grad.sum(axis=tuple(range(grad.ndim - out[self.index].ndim)))
        keepdims = tuple(n for (n, i) in enumerate(grad.shape) if i != out[self.index].shape[n])
        if keepdims:
            grad = grad.sum(axis=keepdims, keepdims=True)
        np.add.at(out, self.index, grad)
        self.variables[0].backward(out)

class Where(Operation):
    def __call__(self, a, b, *, condition):
        self.variables = (a, b)
        self.condition = np.asarray(condition, dtype=bool)
        return np.where(condition, a.data, b.data)

    def backward(self, grad, **kwargs):
        self.variables[0].backward(np.where(self.condition, grad, 0))
        self.variables[1].backward(np.where(~self.condition, grad, 0))

