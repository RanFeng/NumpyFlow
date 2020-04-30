__all__ = ['Operation']

class Operation:
    """
    所有Op的基类

    testLevel 表明当前类的测试等级，
    0表示已经经过大量测试，
    1表示已经经过中等数量测试，
    2表示经过少量无极端示例测试，
    3表示未经过测试
    4表示不够信任
    """
    testLevel = 0
    def __call__(self, *input_vars):
        self.variables = input_vars
        raise NotImplementedError

    def broadcastable(self, grad, ashape):
        """
        保证传递的梯度shape一致，用于兼容广播机制的反向传播
        :param grad:
        :param ashape:
        :return:
        """
        if grad.shape == ashape:
            return grad
        grad_bak = grad.sum(axis=tuple(range(grad.ndim - len(ashape))))
        # print("g", grad_bak.shape, ashape)
        keepdims = tuple(n for (n, i) in enumerate(grad_bak.shape) if i != ashape[n])
        if keepdims:
            grad_bak = grad_bak.sum(axis=keepdims, keepdims=True)
        # print("g", grad_bak.shape, ashape)
        return grad_bak

    def backward(self, grad, **kwargs):
        raise NotImplementedError
