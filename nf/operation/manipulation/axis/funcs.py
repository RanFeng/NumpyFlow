from nf.tensor import Tensor

from .ops import *

__all__ = ["permute","transpose", "swapaxes"]


def permute(a, *axes, requires_grad=False):
    """
    重新排列Tensor的各个维度，等同于numpy中的np.transpose操作
    """
    if not axes:
        axes = None
    return Tensor._op(Permute, a, op_args=(axes,), requires_grad=requires_grad)

def transpose(a, *axes, requires_grad=False):
    """
    转置矩阵，目前适用二维Tensor
    :param a:
    :param axes:
    :param requires_grad:
    :return:
    """
    if(a.ndim < 2):
        raise NotImplemented("此处应当自动扩维，但是还未实现")
    alist = list(range(a.ndim))
    if not axes:    # 若不指定axes，则默认置换最后两个维度
        axes = [a.ndim-1, a.ndim-2]
    alist[axes[0]], alist[axes[1]] = alist[axes[1]], alist[axes[0]],
    return Tensor._op(Permute, a, op_args=(alist,), requires_grad=requires_grad)


def swapaxes(a, axis1, axis2, requires_grad=False):
    """
    交换Tensor的两个维度
    """
    return Tensor._op(SwapAxes, a, op_args=(axis1, axis2), requires_grad=requires_grad)
