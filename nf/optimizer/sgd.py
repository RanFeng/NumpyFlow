from .optimizer import Optimizer


class SGD(Optimizer):
    r"""
    随机梯度下降优化器，支持momentum和nesterov。
    """

    def __init__(self, params, lr=1.e-5, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False):
        if lr < 0.0:
            raise ValueError("请输入正确的learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("请输入正确的momentum: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("请输入正确的weight decay: {}".format(weight_decay))
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov 动量必须要提供动量值")

        self.lr = lr
        self.momentum = momentum
        self.dampening = dampening
        self.weight_decay = weight_decay
        self.nesterov = nesterov
        super(SGD, self).__init__(params)

    def __getstate__(self):
        return {
            'lr': self.lr,
            'momentum': self.momentum,
            'dampening': self.dampening,
            'weight_decay': self.weight_decay,
            'nesterov': self.nesterov,
            'grad_last': self.grad_last,
            'param_groups': self.param_groups,
        }

    def step(self):
        for pid, para in enumerate(self.param_groups):
            d_p = para.grad
            if d_p is None:
                continue
            if self.weight_decay != 0:
                d_p[:] += self.weight_decay * para.data
            if self.momentum != 0:
                if self.grad_last[pid] is None:
                    buf = self.grad_last[pid] = d_p.copy()
                else:
                    buf = self.grad_last[pid]
                    buf *= self.momentum
                    buf += (1 - self.dampening) * d_p
                if self.nesterov:
                    d_p += self.momentum * buf
                else:
                    d_p = buf
            # print(p.data, p.grad)
            para.data[:] += -self.lr * d_p
            # print(p.data, p.grad)

