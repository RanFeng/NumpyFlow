import nf
from .ops.activation import *
from .ops.conv import *
# from .ops.batchnorm import *

from nf import Tensor
import numpy as np

# batchnorm modules


# conv modules

def pad(a, expanded_padding, mode='zeros'):
    return Tensor._op(Pad, a, op_args=(expanded_padding, mode))

def conv2d(a, weight, bias=None, padding='valid', stride=(1,1), dilation=(1,1), groups=1):
    """
    NCHW
    :param a: NCHW
    :param weight: OIHW
    :param bias:
    :param padding:
    :param stride: sH, sW
    :param dilation: dH, dW
    :param groups:
    :return:
    """
    assert padding in ('valid', 'same', 'full') or isinstance(padding, (tuple,list))
    if isinstance(stride, int):
        stride = (stride, stride)
    assert isinstance(stride, (tuple,list))
    if isinstance(dilation, int):
        dilation = (dilation, dilation)
    assert isinstance(dilation, (tuple,list))
    bs, xch, xh, xw = a.shape
    zch, _, k0, k1 = weight.shape
    if isinstance(padding, (tuple, list)):
        a = Tensor._op(Pad, a, op_args=(padding, 'zeros'))
    if padding is 'same':
        zshape = np.ceil([xh / stride[0], xw / stride[1]]).astype(int)
        if stride[0] < k0:
            ph = (k0-1) * dilation[0] + zshape[0] * stride[0] - xh
        else:
            ph = zshape[0] * stride[0] - xh
        if stride[1] < k1:
            pw = (k1-1) * dilation[1] + zshape[1] * stride[1] - xw
        else:
            pw = zshape[1] * stride[1] - xw
        # padding = (pw//2, (pw+1)//2, ph//2, (ph+1)//2)
        padding = (ph//2, (ph+1)//2, pw//2, (pw+1)//2)
        # print(padding, pw, ph, zshape, xh, xw, stride)
        a = Tensor._op(Pad, a, op_args=(padding, 'zeros'))
    out = Tensor._op(Conv2d, a, weight, op_args=(stride, dilation))
    if bias is not None:
        out = out + bias
    return out

def max_pool2d(a, pool_size=(2, 2), stride=(2, 2), padding='valid'):
    """
    :param a:
    :param pool_size:
    :param stride:
    :param padding:
    :return:
    """
    assert padding in ('valid', 'same', 'full') or isinstance(padding, (tuple,list))
    if isinstance(stride, int):
        stride = (stride, stride)
    assert isinstance(stride, (tuple,list))
    if isinstance(pool_size, int):
        pool_size = (pool_size, pool_size)
    assert isinstance(pool_size, (tuple,list))
    bs, xch, xh, xw = a.shape
    if isinstance(padding, (tuple, list)):
        a = Tensor._op(Pad, a, op_args=(padding, 'zeros'))
    # if padding is 'same':
    #     zshape = np.ceil([xh / stride[0], xw / stride[1]]).astype(int)
    #     if stride[0] < pool_size[0]:
    #         ph = (pool_size[0]-1) * dilation[0] + zshape[0] * stride[0] - xh
    #     else:
    #         ph = zshape[0] * stride[0] - xh
    #     if stride[1] < pool_size[1]:
    #         pw = (pool_size[1]-1) * dilation[1] + zshape[1] * stride[1] - xw
    #     else:
    #         pw = zshape[1] * stride[1] - xw
    #     # padding = (pw//2, (pw+1)//2, ph//2, (ph+1)//2)
    #     padding = (ph//2, (ph+1)//2, pw//2, (pw+1)//2)
    #     print(padding, pw, ph, zshape, xh, xw, stride)
    #     a = Tensor._op(Pad, a, op_args=(padding, 'zeros'))
    out = Tensor._op(MaxPool2d, a, op_args=(pool_size, stride, ))
    # return out
    return out



# activation modules
def relu(a, inplace=False):
    return Tensor._op(ReLU, a)

def sigmoid(a):
    return Tensor._op(Sigmoid, a)

def softmax(a, dim = None):
    if dim is None:
        dim = -1
    return Tensor._op(Softmax, a, op_args=(dim,))

