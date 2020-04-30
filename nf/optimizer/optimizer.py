import nf

class Optimizer(object):
    r"""
    所有优化器的基类
    """

    def __init__(self, params):
        if isinstance(params, nf.Tensor):
            raise TypeError("待优化参数必须是nf.Tensor类，不应当是：" + type(params))

        self.param_groups = []
        params = list(params)
        if len(params) == 0:
            raise ValueError("待优化参数为空")
        self.param_groups += params
        self.grad_last = [None, ] * len(self.param_groups)

    def __getstate__(self):
        return {
            'grad_last': self.grad_last,
            'param_groups': self.param_groups,
        }

    def __setstate__(self, state):
        self.__dict__.update(state)

    def zero_grad(self):
        r"""
        将所有参数的梯度清零，准备下一次反向传播
        :return:
        """
        for para in self.param_groups:
            # print("gg",para.requires_grad, para.grad)
            if para.grad is not None:
                para.grad.fill(0.0)

    def step(self):
        raise NotImplementedError
