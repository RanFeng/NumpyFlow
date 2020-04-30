from nf.tensor import Tensor

from .ops import *

__all__ = [
    "exp",
    "log",
    "log2",
    "log10",
]


def exp(a, requires_grad=False):
    return Tensor._op(Exp, a, requires_grad=requires_grad)

def log(a, requires_grad=False):
    return Tensor._op(Log, a, requires_grad=requires_grad)


def log2(a, requires_grad=False):
    return Tensor._op(Log2, a, requires_grad=requires_grad)


def log10(a, requires_grad=False):
    return Tensor._op(Log10, a, requires_grad=requires_grad)
