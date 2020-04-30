import nf
from nf.nn import functional as F
from .module import Module
from . import init

__all__ = ['Conv2d','MaxPool2d']

class _ConvNd(Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 padding='valid', dilation=1, transposed=False, output_padding=None,
                 bias=True, padding_mode='zeros'):
        super(_ConvNd, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.transposed = transposed
        self.output_padding = output_padding
        self.padding_mode = padding_mode
        if transposed:
            self.weight = nf.zeros((in_channels, out_channels, *kernel_size), requires_grad=True)
        else:
            self.weight = nf.zeros((out_channels, in_channels, *kernel_size), requires_grad=True)
        if bias:
            self.bias = nf.zeros((out_channels, ) + (1,) * len(kernel_size), requires_grad=True)  # 2020/4/26 bias的正向和反向传播还未测试
        else:
            self.bias = None
        self.reset_parameters()

    def reset_parameters(self):
        # 5 ** 0.5 = 2.23606798
        init.kaiming_uniform_(self.weight, a=2.23606798)
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / (fan_in ** 0.5)
            init.uniform_(self.bias, -bound, bound)

class _MaxPoolNd(Module):
    def __init__(self, pool_size, stride, padding='valid'):
        super(_MaxPoolNd, self).__init__()
        self.pool_size = pool_size
        self.stride = stride
        self.padding = padding

class Conv2d(_ConvNd):
    # NCHW
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding='valid', dilation=1,
                 bias=True, padding_mode='zeros'):
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        if isinstance(stride, int):
            stride = (stride, stride)
        if isinstance(dilation, int):
            dilation = (dilation, dilation)
        assert len(kernel_size) == 2
        assert len(stride) == 2
        assert len(dilation) == 2
        assert padding in ('valid', 'same', 'full') or isinstance(padding, (tuple,list))
        assert padding_mode in ('zeros')

        super(Conv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, None, bias, padding_mode)


    def forward(self, a):
        return F.conv2d(a, self.weight, self.bias, self.padding,
                        stride=self.stride, dilation=self.dilation)

class MaxPool2d(_MaxPoolNd):
    # NCHW
    def __init__(self, pool_size, stride, padding='valid'):
        if isinstance(pool_size, int):
            pool_size = (pool_size, pool_size)
        if isinstance(stride, int):
            stride = (stride, stride)
        assert len(pool_size) == 2
        assert len(stride) == 2
        assert padding in ('valid', 'same', 'full')

        super(MaxPool2d, self).__init__(pool_size, stride, padding)


    def forward(self, a):
        return F.max_pool2d(a, self.pool_size, self.stride, self.padding)
        # return Functional.linear(input, self.weight, self.bias)


