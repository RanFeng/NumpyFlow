from nf.tensor import *
from nf.operation.creation.funcs import *
from nf.operation.indexing.funcs import *
from nf.operation.manipulation.axis.funcs import *
from nf.operation.manipulation.shape.funcs import *
from nf.operation.math.nondifferentiable import *
from nf.operation.math.arithmetic.funcs import *
from nf.operation.math.exp_log.funcs import *
from nf.operation.math.misc.funcs import *
from nf.operation.math.statistics.funcs import *
from nf.operation.math.linalg.funcs import *



for attr in (
    sum,
    mean,
    std,
    var,
    max,
    min,
    argmax,
    argmin,
    swapaxes,
    permute,
    transpose,
    # moveaxis,
    flatten,
    reshape,
    squeeze,
    dot,
    matmul,
    einsum,
):
    setattr(Tensor, attr.__name__, attr)