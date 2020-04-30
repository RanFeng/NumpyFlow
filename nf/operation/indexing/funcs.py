import numpy as np
from nf.tensor import Tensor
from .ops import *

__all__ = ["where"]


def where(condition, x=None, y=None, requires_grad=False):
    if x is None and y is None:
        if isinstance(condition, Tensor):
            condition = condition.data
        return np.where(condition)

    return Tensor._op(Where, x, y, op_kwargs=dict(condition=condition), requires_grad=requires_grad)
