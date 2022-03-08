import numpy as np
from typing import Optional, Set, Type, Union

from .op import Operation
from .operation.math.arithmetic.ops import *
from .operation.math.linalg.ops import *
from .operation.manipulation.shape.ops import *
from .operation.manipulation.axis.ops import *
from .operation.indexing.ops import *

np.set_printoptions(suppress=True,linewidth=300)

__all__ = ["Tensor"]

class Tensor:
    def __init__(self, data=None, *, requires_grad=False, creator=None):
        assert isinstance(requires_grad, bool)
        assert isinstance(creator, (Operation, None.__class__))
        self.data = None
        if isinstance(data, (int, float, bool)):
            data = [data]
        if isinstance(data, (list, tuple)):
            data = np.array(data)
        if isinstance(data, np.ndarray):
            self.data = data.copy()
        elif isinstance(data, Tensor):
            raise ValueError("输入的是 Tensor")
        else:
            raise ValueError("输入类型未知", type(data), data)
        self.requires_grad = requires_grad

        if creator is None:
            creator = Assign()
            creator(self)   # 看似没有用，但是可以用来在计算图的可视化，可视化Assign节点
        self.creator = creator
        self.grad = None

    @classmethod
    def _op(cls,
            Op: Type[Operation],
            *input_vars,
            op_args=None,
            op_kwargs=None,
            requires_grad=False
            ):
        if op_args is None:
            op_args = tuple()

        if op_kwargs is None:
            op_kwargs = dict()

        tensor_vars = tuple(
            cls(var, requires_grad=False) if not isinstance(var, cls) else var
            for var in input_vars
        )
        requires_grad = requires_grad or any(var.requires_grad for var in tensor_vars)
        f = Op()
        op_out = f(*tensor_vars, *op_args, **op_kwargs)
        return cls(op_out, requires_grad=requires_grad, creator=f)


    def __add__(self, other):
        return self._op(Add, self, other)

    def __radd__(self, other):
        return self._op(Add, other, self)

    def __mul__(self, other):
        return self._op(Multiply, self, other)

    def __rmul__(self, other):
        return self._op(Multiply, other, self)

    def __sub__(self, other):
        return self._op(Subtract, self, other)

    def __rsub__(self, other):
        return self._op(Subtract, other, self)

    def __truediv__(self, other):
        return self._op(Divide, self, other)

    def __rtruediv__(self, other):
        return self._op(Divide, other, self)

    def __pow__(self, other):
        return self._op(Power, self, other)

    def __rpow__(self, other):
        return self._op(Power, other, self)

    def __neg__(self):
        return self._op(Negative, self)

    def __pos__(self):
        return self._op(Positive, self)


    def __matmul__(self, other):
        return self._op(MatMul, self, other)

    def __rmatmul__(self, other):
        return self._op(MatMul, other, self)

    def backward(self, grad=None):
        if not self.requires_grad:
            return
        if grad is None:
            grad = np.ones_like(self.data, dtype=np.float64)
        if isinstance(grad, Tensor):
            grad = grad.data
        if isinstance(self.creator, Assign) or True: # 计算原子
            if self.grad is None:
                self.grad = np.zeros_like(self.data, dtype=np.float64)

            try:    # try except 成本比if低？那就不亏
                self.grad += grad
            except ValueError:  # self.grad.shape 长度或大小小于 grad.shape，用于适应广播机制
                grad_bak = grad.sum(axis=tuple(range(grad.ndim - self.grad.ndim)))
                keepdims = tuple(n for (n, i) in enumerate(grad_bak.shape) if i != self.grad.shape[n])
                if keepdims:
                    grad_bak = grad_bak.sum(axis=keepdims, keepdims=True)
                self.grad += grad_bak
        self.creator.backward(grad)

    def numpy(self):
        return self.data.copy()

    def copy(self):
        return self.__copy__()

    def __copy__(self):
        """
        复制当前Tensor的grad、data、requires_grad，设定creator=None
        如果当前的Tensor没有梯度，则梯度为None
        Returns
        -------
        Tensor
        """
        copy = Tensor(np.copy(self.data),requires_grad=self.requires_grad, creator=None)
        try:
            copy.grad[:] = self.grad[:]    # 尽量复制梯度
        except:
            pass
        return copy

    def copy_(self, other):
        assert isinstance(other, self.__class__)
        self.data[:] = other.data
        try:
            self.grad[:] = other.grad[:]    # 尽量复制梯度
        except:
            pass



    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return "Tensor with shape: {}\n{}".format(self.shape, self.data)

    def __len__(self):
        return len(self.data)

    def __contains__(self, item):
        return self.data.__contains__(item)

    def __getitem__(self, item):
        return self._op(GetItem, self, op_args=(item,))

    def __setitem__(self, key, value):
        raise NotImplemented("还没实现，好像很多的样子，下次一定")

    def item(self):
        """
        用来将Tensor转化成一般的python类型，返回值不支持求导
        Returns
        -------
        z : 一般的python类型，如float等等

        Examples
        --------
        >>> import nf
        >>> x = Tensor([22.2])
        >>> x.item()
        22.2
        >>> type(x.item())
        float
        """
        if self.size > 1:
            raise ValueError("不能转化size大于1的Tensor")
        return self.data.item()

    def __float__(self):
        if self.size > 1:
            raise TypeError("不能转化size大于1的Tensor")
        return float(self.data)

    def __int__(self):
        if self.size > 1:
            raise TypeError("不能转化size大于1的Tensor")
        return int(self.data)

    @property
    def size(self):
        """
        返回一个int值，表示当前Tensor的data的size
        Returns
        -------
        int

        Examples
        --------
        >>> import nf
        >>> x = nf.zeros((3, 5, 2))  # creates a tensor with 3x5x2 (= 30) elements
        >>> x.size
        30
        """
        return self.data.size

    @property
    def ndim(self):
        """
        返回当前Tensor的维度

        Returns
        -------
        int

        Examples
        --------
        >>> import nf
        >>> x = nf.ones_like((2,3,4,1,4))
        >>> x.ndim
        5
        """
        return self.data.ndim

    @property
    def dtype(self):
        """
        返回当前Tensor的数组类型，也就是numpy中的类型

        Returns
        -------
        numpy dtype object

        <type 'numpy.dtype'>"""
        return self.data.dtype

    @property
    def shape(self):
        """
        返回当前Tensor的shape

        Returns
        -------
        Tuple[int, ...]

        Examples
        --------
        >>> import nf
        >>> x = nf.Tensor([1, 2, 3, 4])  # axis-0 has size 4
        >>> x.shape
        (4,)
        >>> y = nf.Tensor([[1, 2, 3],    # axis-0 has size 2, axis-1 has size 3
        ...                [4, 5, 6]])
        >>> y.shape
        (2, 3)

        """
        return self.data.shape

    @property
    def T(self):
        """
        返回当前Tensor的转置，在此与numpy一致，是一个属性，
        如果当前Tensor.ndim > 2，返回的就是整个数组的转置。

        Returns
        -------
        Tensor

        Examples
        --------
        >>> import nf
        >>> y = nf.Tensor([[1, 2, 3],
        ...                [4, 5, 6]])
        >>> y.T()
        Tensor([[1, 4],
                [2, 5],
                [3, 6]])
        """
        return self._op(Permute, self)

