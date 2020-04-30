import numpy as np
from numpy.core.einsumfunc import _parse_einsum_input


from nf.tensor import Tensor


from .ops import *

__all__ = ["dot","matmul", "einsum"]


def matmul(a, b, requires_grad=False):
    return Tensor._op(MatMul, a, b, requires_grad=requires_grad)

dot = matmul

def einsum(*operands, optimize=False, requires_grad=False):
    # 这段没有验证过，直接超过来的
    operands = list(operands)
    if isinstance(operands[0], str):
        # operands form: "ijk, ijk", x, y
        variables = operands[1:]
        if any(isinstance(i, Tensor) for i in operands):
            operands[1:] = (
                var.data if isinstance(var, Tensor) else var for var in operands[1:]
            )
    else:
        # operands form: op0, sublist0, op1, sublist1, ..., [sublistout]
        end = -1 if len(operands) % 2 else None  # -1 if sublistout is included
        variables = operands[:end:2]
        if any(isinstance(i, Tensor) for i in operands):
            operands[:end:2] = (
                var.data if isinstance(var, Tensor) else var for var in operands[:end:2]
            )

    in_lbls, out_lbls, _ = _parse_einsum_input(operands)
    return Tensor._op(
        EinSum,
        *variables,
        op_kwargs=dict(in_lbls=in_lbls, out_lbls=out_lbls, optimize=optimize),
        requires_grad=requires_grad
    )

