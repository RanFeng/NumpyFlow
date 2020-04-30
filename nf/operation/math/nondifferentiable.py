import numpy as np

from nf.tensor import Tensor

__all__ = ["argmin", "argmax"]


def argmax(a, axis=None, out=None):
    a = a.data if isinstance(a, Tensor) else a
    return np.argmax(a, axis, out)


def argmin(a, axis=None, out=None):
    a = a.data if isinstance(a, Tensor) else a
    return np.argmin(a, axis, out)
