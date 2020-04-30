from nf.tensor import Tensor

from .ops import *

__all__ = ["flatten","reshape", "squeeze", "expand_dims"]


def flatten(a, requires_grad=False):
    return Tensor._op(Flatten, a, requires_grad=requires_grad)

def reshape(a, *newshape, requires_grad=False):
    if not newshape:
        raise TypeError("reshape() takes at least 1 argument (0 given)")
    return Tensor._op(Reshape, a, op_args=(newshape,), requires_grad=requires_grad)


def squeeze(a, axis=None, requires_grad=False):
    return Tensor._op(Squeeze, a, op_args=(axis,), requires_grad=requires_grad)


def expand_dims(a, axis, requires_grad=False):
    return Tensor._op(ExpandDims, a, op_args=(axis,), requires_grad=requires_grad)
