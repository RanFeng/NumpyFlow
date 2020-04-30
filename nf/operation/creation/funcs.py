import numpy as np
from nf.tensor import Tensor


__all__ = [
    "empty",
    "empty_like",
    "ones",
    "ones_like",
    "zeros",
    "zeros_like",
    "rands"
]


def empty(shape, dtype=np.float32, requires_grad=False):
    """
    将 np.empty(shape, dtype) 填入Tensor的数据中
    :return
    返回一个给定shape和dtype的Tensor
    """
    return Tensor(np.empty(shape, dtype), requires_grad=requires_grad)


def empty_like(other, dtype=None, requires_grad=False):
    """
    将 np.empty_like(other, dtype) 填入Tensor的数据中
    :return
    返回一个与目标形状和类型一致的Tensor
    """
    if isinstance(other, Tensor):
        other = other.data

    return Tensor(np.empty_like(other, dtype), requires_grad=requires_grad)


def ones(shape, dtype=np.float32, requires_grad=False):
    """
    将 np.ones(shape, dtype) 填入Tensor的数据中
    :return
    返回一个给定shape和dtype的Tensor
    """
    return Tensor(np.ones(shape, dtype), requires_grad=requires_grad)


def ones_like(other, dtype=None, requires_grad=False):
    """
    将 np.ones_like(other, dtype) 填入Tensor的数据中
    :return
    返回一个与目标形状和类型一致的Tensor
    """
    if isinstance(other, Tensor):
        other = other.data

    return Tensor(np.ones_like(other, dtype), requires_grad=requires_grad)


def zeros(shape, dtype=np.float32, requires_grad=False):
    """
    将 np.zeros(shape, dtype) 填入Tensor的数据中
    :return
    返回一个给定shape和dtype的Tensor
    """
    return Tensor(np.zeros(shape, dtype), requires_grad=requires_grad)


def zeros_like(other, dtype=None, requires_grad=False):
    """
    将 np.zeros_like(other, dtype) 填入Tensor的数据中
    :return
    返回一个与目标形状和类型一致的Tensor
    """
    if isinstance(other, Tensor):
        other = other.data

    return Tensor(np.zeros_like(other, dtype), requires_grad=requires_grad)

def rands(shape, requires_grad=False):
    """
    将 np.random.random(shape) 填入Tensor的数据中
    :return
    返回一个给定shape的随机Tensor
    """
    return Tensor(np.random.random(shape), requires_grad=requires_grad)








