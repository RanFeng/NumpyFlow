from nf.tensor import Tensor

from .ops import *

__all__ = [
    "add",
    "divide",
    "multiply",
    "negative",
    "positive",
    "power",
    "subtract",
]


def add(a, b, requires_grad=False):
    return Tensor._op(Add, a, b, requires_grad=requires_grad)


def subtract(a, b, requires_grad=False):
    return Tensor._op(Subtract, a, b, requires_grad=requires_grad)


def divide(a, b, requires_grad=False):
    return Tensor._op(Divide, a, b, requires_grad=requires_grad)


def power(a, b, requires_grad=False):
    return Tensor._op(Power, a, b, requires_grad=requires_grad)


def multiply(a, b, requires_grad=False):
    return Tensor._op(Multiply, a, b, requires_grad=requires_grad)

def positive(a, requires_grad=False):
    return Tensor._op(Positive, a, requires_grad=requires_grad)

def negative(a, requires_grad=False):
    return Tensor._op(Negative, a, requires_grad=requires_grad)
