from nf.op import Operation
import numpy as np

__all__ = ['ReLU','Sigmoid','Softmax']





class ReLU(Operation):
    def __call__(self, a):
        """ Performs 'add' forward-pass: f(a,b) -> a + b

            Parameters
            ----------
            a : Tensor
            b : Tensor

            Returns
            -------
            out : numpy.ndarray """

        self.variables = (a,)
        self.mask = a.data>0.0
        out = a.data * self.mask
        return out

    def backward(self, grad, **kwargs):
        a = self.variables[0]
        self.variables[0].backward(grad * self.mask)

class Sigmoid(Operation):
    testLevel = 2

    def __call__(self, a):
        self.variables = (a,)
        out = 1 / (1 + np.exp(-a.data))
        self.out = out
        return out

    def backward(self, grad, **kwargs):
        a = self.variables[0]
        # f = 1 / (1 + np.exp(-a.data))
        f = self.out * (1 - self.out)
        self.variables[0].backward(grad * f)

class Softmax(Operation):
    def __call__(self, a, dim):
        """ Performs 'add' forward-pass: f(a,b) -> a + b

            Parameters
            ----------
            a : Tensor
            b : Tensor

            Returns
            -------
            out : numpy.ndarray """

        self.variables = (a,)
        self.dim = dim
        # print(a.data)
        out = np.exp(a.data - np.max(a.data, axis=self.dim, keepdims=True))
        out /= np.sum(out, axis=self.dim, keepdims=True)
        self.out = out
        return out


    def backward(self, grad, **kwargs):
        a = self.variables[0]
        grad *= self.out
        p = grad.sum(axis=self.dim, keepdims=True)
        grad -= self.out * p
        self.variables[0].backward(grad)


