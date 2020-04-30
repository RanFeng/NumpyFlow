from nf.op import Operation
import numpy as np

__all__ = ['Assign',
           'Add',
           'Multiply',
           'Subtract',
           'Divide',
           'Negative',
           'Positive',
           'Power']



class Assign(Operation):
    def __call__(self, a):
        self.variables = (a)
        return a

    def backward(self, grad, **kwargs):
        return None


class Add(Operation):
    def __call__(self, a, b):
        self.variables = (a, b)
        out = a.data + b.data
        return out


    def backward(self, grad, **kwargs):
        self.variables[0].backward(grad)
        self.variables[1].backward(grad)


class Multiply(Operation):
    def __call__(self, a, b):
        self.variables = (a, b)
        out = a.data * b.data
        return out

    def backward(self, grad, **kwargs):
        a, b = self.variables
        self.variables[0].backward(b.data * grad)
        self.variables[1].backward(a.data * grad)


class Subtract(Operation):
    def __call__(self, a, b):
        self.variables = (a, b)
        out = a.data - b.data
        return out

    def backward(self, grad,**kwargs):
        self.variables[0].backward(grad)
        self.variables[1].backward(-grad)


class Divide(Operation):
    def __call__(self, a, b):
        self.variables = (a, b)
        self.out = a.data / b.data
        return self.out

    def backward(self, grad, **kwargs):
        a, b = self.variables
        p = grad / b.data
        self.variables[0].backward(p)
        self.variables[1].backward(-p * self.out)


class Negative(Operation):

    def __call__(self, a, where=True):
        self.variables = (a,)
        return -a.data

    def backward(self, grad, **kwargs):
        self.variables[0].backward(-grad)


class Positive(Operation):

    def __call__(self, a):
        self.variables = (a,)
        return np.positive(a.data)

    def backward(self, grad,**kwargs):
        self.variables[0].backward(np.positive(grad))



class Power(Operation):
    def __call__(self, a, b):
        self.variables = (a, b)
        out = a.data ** b.data
        return out

    def backward(self, grad,**kwargs):
        a, b = self.variables
        x, y = a.data, b.data
        self.variables[0].backward(grad * y * (x ** np.where(y, (y - 1), 1)))
        self.variables[1].backward(grad * (x ** y) * np.log(np.where(x, x, 1)))

