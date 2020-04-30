from nf.op import Operation
import numpy as np


__all__ = ['Flatten','Reshape','Squeeze','ExpandDims']


class Flatten(Operation):
    testLevel = 2
    def __call__(self, a):
        self.variables = (a,)
        return a.data.flatten(order="C")

    def backward(self, grad, **kwargs):
        a = self.variables[0]
        self.variables[0].backward(grad.reshape(*a.data.shape))



class Reshape(Operation):
    testLevel = 2
    def __call__(self, a, shape):
        self.variables = (a,)
        if shape is not None and hasattr(shape, "__iter__"):
            shape = shape[0]
        self.shape = shape
        return a.data.reshape(shape)

    def backward(self, grad, **kwargs):
        a = self.variables[0]
        try:
            grad = grad.reshape(*a.shape)
        except ValueError:
            grad = self.broadcastable(grad, self.shape)
            grad = grad.reshape(*a.shape)
        self.variables[0].backward(grad)


class Squeeze(Operation):
    testLevel = 4
    def __call__(self, a, axis):
        self.variables = (a,)
        return np.squeeze(a.data, axis=axis)

    def backward(self, grad, **kwargs):
        a = self.variables[0]
        self.variables[0].backward(grad.reshape(*a.shape))



class ExpandDims(Operation):
    testLevel = 4
    def __call__(self, a, axis):
        self.variables = (a,)
        out = np.expand_dims(a.data, axis=axis)
        self.outshape = out
        return

    def backward(self, grad, **kwargs):
        a = self.variables[0]
        try:
            grad = grad.reshape(*a.shape)
        except ValueError:
            grad = self.broadcastable(grad, self.outshape)
            grad = grad.reshape(*a.shape)
        self.variables[0].backward(grad)
        # grad = self.broadcastable(grad, a.shape)
        # self.variables[0].backward(grad)





















