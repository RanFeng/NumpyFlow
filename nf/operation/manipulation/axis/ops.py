from nf.op import Operation
import numpy as np

__all__ = ['Permute', "SwapAxes"]

class Permute(Operation):
    testLevel = 4
    def __call__(self, a, axes=None):

        self.variables = (a,)
        if axes is not None:
            self.axes = tuple(axis % a.ndim for axis in axes)
        else:
            self.axes = tuple(range(a.ndim)[::-1])
        # print(self.axes,a.data.shape)

        return np.transpose(a.data, self.axes)

    def backward(self, grad, **kwargs):
        a = self.variables[0]
        # print(grad.shape, a.shape, self.axes)
        try:
            grad = grad.transpose(np.argsort(self.axes))
        except ValueError:
            grad = self.broadcastable(grad, self.variables[0].shape[::-1])
            grad = grad.transpose(np.argsort(self.axes))
        # print(grad.shape, a.shape, self.axes)
        self.variables[0].backward(grad)

class SwapAxes(Operation):
    def __call__(self, a, axis1, axis2):
        self.variables = (a,)
        self.axis1 = axis1
        self.axis2 = axis2
        return np.swapaxes(a.data, axis1, axis2)

    def backward(self, grad, **kwargs):
        self.variables[0].backward(grad.swapaxes(self.axis2, self.axis1))
















