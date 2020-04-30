from nf.tensor import Tensor

from .ops import *
__all__ = ["abs", "absolute", "cbrt", "clip", "maximum", "minimum"]


def abs(a, requires_grad=False):
    return Tensor._op(Abs, a, requires_grad=requires_grad)


absolute = abs


def cbrt(a, requires_grad=False):
    return Tensor._op(Cbrt, a, requires_grad=requires_grad)


def maximum(a, b, requires_grad=False):
    return Tensor._op(Maximum, a, b, requires_grad=requires_grad)


def minimum(a, b, requires_grad=False):
    return Tensor._op(Minimum, a, b, requires_grad=requires_grad)


def clip(a, a_min, a_max, requires_grad=False):
    if a_min is None and a_max is None:
        raise ValueError("`a_min` 与 `a_max` 不能都为空")

    if a_min is not None:
        a = maximum(a_min, a, requires_grad=requires_grad)

    if a_max is not None:
        a = minimum(a_max, a, requires_grad=requires_grad)

    return a

