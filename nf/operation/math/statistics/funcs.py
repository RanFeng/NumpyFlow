from nf.tensor import Tensor

from .ops import *
__all__ = [
    "sum",
    "mean",
    "var",
    "std",
    "amax",
    "amin",
    "max",
    "min",
]


def sum(x, axis=None, keepdims=False, requires_grad=False):
    return Tensor._op(Sum, x, op_args=(axis, keepdims), requires_grad=requires_grad)


def mean(x, axis=None, keepdims=False, requires_grad=False):
    return Tensor._op(Mean, x, op_args=(axis, keepdims), requires_grad=requires_grad)


def var(x, axis=None, ddof=0, keepdims=False, requires_grad=False):

    return Tensor._op(
        Variance,
        x,
        op_kwargs=dict(axis=axis, keepdims=keepdims, ddof=ddof),
        requires_grad=requires_grad,
    )


def std(x, axis=None, ddof=0, keepdims=False, requires_grad=False):
    return Tensor._op(
        StdDev,
        x,
        op_kwargs=dict(axis=axis, keepdims=keepdims, ddof=ddof),
        requires_grad=requires_grad,
    )


def max(x, axis=None, keepdims=False, requires_grad=False):
    return Tensor._op(
        MaxMin,
        x,
        op_kwargs=dict(axis=axis, keepdims=keepdims, maxmin="max"),
        requires_grad=requires_grad,
    )


def min(x, axis=None, keepdims=False, requires_grad=False):
    return Tensor._op(
        MaxMin,
        x,
        op_kwargs=dict(axis=axis, keepdims=keepdims, maxmin="min"),
        requires_grad=requires_grad,
    )


# aliases
amin = min
amax = max

