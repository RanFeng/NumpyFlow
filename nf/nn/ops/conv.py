from nf.op import Operation
import numpy as np
from nf.backend import numcy

__all__ = ['Conv2d','Pad','MaxPool2d']


def _pad(x, ep):
    # 保持跟numpy的pad接口一致
    xshape = np.array(x.shape)
    ep = np.array(ep)
    es = ep.sum(1)
    pad_array = np.zeros(xshape+es)
    ep = tuple(slice(si[0], -si[1]) if si[1] > 0 else slice(None, None) for si in ep)
    pad_array[ep] = x[:]
    return pad_array


def im2bchwkl(x, ksize, stride=(1, 1), dilation=(1, 1), writeable=False):
    """
    :param x: [bs, xch, x1, x2]
    :param ksize: [k1, k2]
    :param stride: [s1, s2]
    :param dilation: [d1, d2]
    :param writeable:
    :return: [bs, xch, z0, z1, k0, k1]
    """
    bs, xch, x1, x2 = x.shape
    H = (x1-(dilation[0]*(ksize[0]-1)+1))//(stride[0])+1
    W = (x2-(dilation[1]*(ksize[1]-1)+1))//(stride[1])+1
    _s = np.array([xch * x1 * x2, x1 * x2, x2 * stride[0], stride[1], x2 * dilation[0], dilation[1]]) * x.itemsize
    return np.lib.stride_tricks.as_strided(x,
                                           (bs, xch, H, W, ksize[0], ksize[1]),
                                           _s,
                                           writeable=writeable,)


def _conv2d_fw(x, w, stride, dilation):
    x = im2bchwkl(x, w.shape[-2:], stride, dilation)  # [bs, xch, z0, z1, k0, k1]
    # out = np.tensordot(x, weight.data, ((1, 4, 5), (1, 2, 3))) # [bs, z0, z1, zch]
    # out = out.transpose(0, 3, 1, 2)
    # if self.path is None:
    #     t1 = time()
    #     self.path = np.einsum_path('bchwkl,zckl->bzhw',x, weight.data, optimize=True)[1]
    #     print(time()-t1)
    # print(a.shape, x.shape, weight.shape)
    out = np.einsum('bchwkl,zckl->bzhw', x, w, optimize=True, order='C')
    return out

def _conv2d_bpl(xt, w, s, d):
    """
    xt is result tensor grad, w is weight tensor, zt is input gradient
    在卷积层前向传播时，有：
        z = (x - d * (k - 1) - 1 + s) // s
    在反向传播求上一层梯度时，我们先推导，在前向传播时，上式的x一般不能使得完成整除，
    必然有后面y位被舍弃，未参加卷积，故上式可改成 ：
        z = (x - d * (k - 1) - 1 + s - y) / s
    反向传播时，我们期望得到大小为x的矩阵，但是由于y的存在，y的那部分的梯度应当为0，
    所以我们在反卷积的时候，只能求出 (x-y) 部分的导数。
        xt = z
        经过dilate和padding之后
        xt = (z - 1) * s + 1 + 2 * (k - 1)
    我们先以特殊情况 d = 1 来计算：
        zt = xt - k + 1
           = z * s - s + 2 * k - 1 - k + 1
           = x - y - (k - 1) - 1 + s - s + k
           = x - y
    至此，求得导数是 zt = x - y 大小的矩阵
    :param x: ( bs, zch, x0, x1)
    :param w: (zch, xch, k0, k1)
    :param s: (s1, s2)
    :return: (bs, xch, z0, z1)
    """
    bs , zch, x0, x1 = xt.shape
    zch, xch, k0, k1 = w.shape

    # dilate a numpy array
    ph = 2*d[0]*(k0-1) + s[0] * (x0-1) + 1
    pw = 2*d[1]*(k1-1) + s[1] * (x1-1) + 1
    dil_array = np.zeros((bs, zch, ph, pw))
    if k0 is 1:
        slice_h = slice(None, None, s[0])
    else:
        slice_h = slice(d[0]*(k0-1),-d[0]*(k0-1),s[0])
    if k1 is 1:
        slice_w = slice(None, None, s[1])
    else:
        slice_w = slice(d[1]*(k1-1),-d[1]*(k1-1),s[1])
    dil_array[:,:,slice_h,slice_w] = xt[:,:]
    x = im2bchwkl(dil_array, w.shape[-2:], (1,1), d)  # [bs, xch, z0, z1, k0, k1]
    zt = np.einsum('bchwkl,czkl->bzhw', x, w[:,:,::-1,::-1], optimize=True, order='C')
    # w = w[:, :, ::-1, ::-1].transpose(1, 0, 2, 3)  # (xch, zch, k0, k1)
    # zt = _conv2d_fw(dil_array, w, (1,1), d)
    return zt

def _conv2d_bpw(x, z, s, d):
    """
    x is input tensor, z is output gradient, w is weight tensor gradient
    以 z 作为卷积核，对x做valid卷积操作，步长为1，空洞为s
    有 z = (x - d * (k - 1) - 1 + s - y) / s
       kt = (z - 1) * s + 1
          = x - y - (k - 1)
       zt = x - kt + 1
          = x - x + y + k - 1 + 1
          = k + y
    最终得到的 zt 需要将后面的 y 位舍弃
    :param x: (bs, xch, x0, x1)
    :param z: (bs, zch, z0, z1)
    :param s: (s1, s2)
    :return: (zsh, xch, k0, k1)
    """
    x = im2bchwkl(x, z.shape[-2:], d, s)  # [bs, xch, z0, z1, k0, k1]
    zt = np.einsum('bchwkl,bzkl->zchw', x, z, optimize=True, order='C')
    return zt

from time import time
class Conv2d(Operation):
    path = None
    def __call__(self, a, weight, stride=(1,1), dilation=(1,1)):
        """
        NCHW
        :param a: (bs, xch, xh, xw)
        :param weight: (zch, xch, k1, k2)
        :return: z: (bs, zch, z1, z2)
        """
        self.variables = (a, weight)
        self.stride = stride
        self.dilation = dilation
        out = _conv2d_fw(a.data, weight.data, stride, dilation)
        return out

    def backward(self, grad, **kwargs):
        a, b = self.variables
        grada = _conv2d_bpl(xt=grad, w=b.data, s=self.stride, d=self.dilation)

        if grada.shape != a.shape:
            _, _, zh, zw = grada.shape
            zt = np.zeros(a.shape)
            zt[:,:,:zh,:zw] = grada[:,:]
            grada = zt
        # print("bp", grad.shape, grada.shape, a.shape, b.shape)
        self.variables[0].backward(grada)
        # 获得本层的 delta_w
        delta_w = _conv2d_bpw(a.data, grad, self.stride, self.dilation)
        # print("d",delta_w.shape, b.shape)
        # print(delta_w)
        _w1, _w2, _w3, _w4 = b.shape
        delta_w = delta_w[:_w1, :_w2, :_w3, :_w4]
        self.variables[1].backward(delta_w)

class Pad(Operation):
    def __call__(self, a, expanded_padding, mode='zeros'):
        """
        :param a:
        :param expanded_padding:
            注意，torch的padding格式为(a,b,c,d,e,f,...)，其中(a,b)表示0维的前后padding数量，(c,d)表示1维的padding数量，以此类推
            而numpy的padding格式为((a,b),(c,d),(e,f),...)，其中(a,b)表示0维的前后padding数量，(c,d)表示1维的padding数量，以此类推
            torch的padding，如果缺少，比如输入(a,b)，而实际上输入有多个维度，则表示其他维度padding都为0，只有-1维padding(a,b)，
                并且torch强制用户输入的len(expanded_padding) 是偶数个，此处我们与torch保持一致，要求输入为偶数个
            numpy的padding，如果缺少，比如输入((a,b),)，而实际上有多个维度，则表示所有维度padding都为(a,b)
            为了与torch对线，我们此处用numpy的padding实现torch的padding接口。
        :param mode:
        :return:
        """
        assert len(expanded_padding) % 2 == 0
        self.variables = (a,)
        t = a.ndim * 2 - len(expanded_padding)
        self.expanded = (0,) * t
        self.expanded += expanded_padding

        self.expanded = np.array(self.expanded).reshape(-1,2)
        # self.expanded = np.array(self.expanded).reshape(-1,2)[::-1,:]
        # print(self.expanded, a)
        # out = np.pad(a.data, self.expanded)
        out = _pad(a.data, self.expanded)
        return out

    def backward(self, grad, **kwargs):
        a = self.variables[0]
        ashape = a.shape
        ep = tuple(slice(si[0], -si[1]) if si[1] > 0 else slice(None, None) for si in self.expanded)
        grad = grad[ep]
        # print("pad grad", grad.shape,ep)
        self.variables[0].backward(grad)

class MaxPool2d(Operation):
    def __call__(self, a, pool_size=(2, 2), stride=(2, 2)):
        """
        :param a:
        :param pool_size:
        :param strides:
        :param pool_mode: 'max' or 'avg'
        """
        self.variables = (a, )
        self.stride = stride
        self.pool_size = pool_size
        self.block_view = im2bchwkl(a.data, self.pool_size, self.stride, (1, 1), True)
        out = numcy.maxpool2d_fw(self.block_view)
        return out

    def backward(self, grad, **kwargs):
        a = self.variables[0]
        outgrad = np.zeros_like(a.data)
        dst = im2bchwkl(outgrad, self.pool_size, self.stride, (1, 1), True)
        numcy.maxpool2d_bp(self.block_view, dst, grad)
        self.variables[0].backward(outgrad)
