from nf.op import Operation
import numpy as np
from collections.abc import Sequence
from typing import Any
__all__ = ["MaxMin", "Sum", "Mean", "Variance", "StdDev"]


class MaxMin(Operation):
    def __call__(self, a, axis=None, keepdims=False, maxmin=None):

        assert maxmin in ("max", "min"), "Invalid keyword argument"
        op = np.argmax if maxmin == "max" else np.argmin

        # let numpy handle error checking
        np.amax(np.empty([1] * a.ndim), axis=axis, keepdims=keepdims)

        self.variables = (a,)

        if a.ndim == 0:
            return a.data

        if hasattr(axis, "__iter__"):
            assert isinstance(axis, tuple)
            axis = tuple(ax % a.ndim for ax in axis)
            axis = None if len(axis) == a.ndim else tuple(sorted(axis))
        elif axis is not None:
            axis = (axis % a.ndim,)

        self.axis = axis
        self.keepdims = keepdims

        # max(a) -> use argmax
        if self.axis is None:
            self.indices = np.unravel_index(op(a.data), a.shape)
            dat = a.data[self.indices]

        # max(x, axis=i) -> use argmax with specified axis
        elif len(self.axis) == 1:  #
            op_index = op(a.data, axis=self.axis[0])
            self.indices = list(np.indices(op_index.shape))
            self.indices.insert(self.axis[0], op_index)
            self.indices = tuple(self.indices)
            dat = a.data[self.indices]

        # max(x, axis=(i,j,...) ) -> Reshape data to use argmax along trailing axis
        else:
            self.static_ax = tuple(
                sorted(set(range(a.ndim)) - set(self.axis))
            )  # non-reduced axes (m, n, ..)
            self.to_trans = self.static_ax + self.axis  # (m, n, ..., i, j, ...)
            self.from_trans = tuple(np.argsort(self.to_trans))
            outshape = tuple(a.shape[i] for i in self.static_ax)

            z = a.data.transpose(*self.to_trans).reshape(
                *outshape, -1
            )  # (m, n, ..., i*j*[...])

            k = op(z, axis=-1)
            self.indices = tuple(i for i in np.indices(k.shape))
            self.indices += (k,)
            self.tmp_grad_shape = z.shape
            z = z[self.indices]

            dat = z.reshape(outshape)  # (m, n, ...)

        if not self.keepdims:
            return dat

        elif self.axis is None:
            keep_index = (np.newaxis,) * a.ndim
        else:
            keep_index = [slice(None)] * a.ndim
            for i in self.axis:
                keep_index[i] = np.newaxis
            keep_index = tuple(keep_index)

        return np.asarray(dat)[keep_index]


    def backward(self, grad, **kwargs):
        a = self.variables[0]
        if a.ndim == 0:
            self.variables[0].backward(grad)
            return

        # normalize shape of grad to be same as when keepdims=False
        if self.keepdims:
            if self.axis is not None:
                reduce = [slice(None)] * a.ndim
                for i in self.axis:
                    reduce[i] = 0
                reduce = tuple(reduce)
            else:
                reduce = (0,) * a.ndim
            grad = grad[reduce]

        # use argmax indices to broadcast grad to correct elements
        if self.axis is None or len(self.axis) == 1:
            out = np.zeros_like(a.data, dtype=float)
            out[self.indices] = grad
        else:
            out = np.zeros(self.tmp_grad_shape, dtype=float)
            out[self.indices] = grad
            shape = tuple(a.shape[i] for i in self.to_trans)
            out = out.reshape(shape).transpose(*self.from_trans)
        self.variables[0].backward(out)


class Sum(Operation):
    testLevel = 2
    def __call__(self, a, axis=None, keepdims=False):
        self.variables = (a,)

        if axis is not None and not hasattr(axis, "__iter__"):
            axis = (axis,)
        if axis is None:
            axis = tuple(i for i in range(len(a.shape)))
        self.axis = axis

        self.keepdims = keepdims
        out = a.data.sum(axis=axis, keepdims=keepdims)
        out = np.array(out)
        self.outshape = out.shape
        return out

    def backward(self, grad, **kwargs):
        a = self.variables[0]
        # print(grad.shape, a.shape, self.axis, self.outshape)
        grad = self.broadcastable(grad, self.outshape)
        # grad = self.broadcastable(grad, a.shape)
        # print(grad.shape, a.shape, self.axis, self.outshape)
        if not self.keepdims:
            index = [slice(None) for i in range(a.ndim)]
            for i in self.axis:
                index[i] = np.newaxis
            grad = grad[tuple(index)]
        grad = np.broadcast_to(grad, a.data.shape).astype(float)
        # print("sum",grad.shape, a.shape, self.axis, self.outshape)
        self.variables[0].backward(grad)

class Mean(Operation):
    testLevel = 2
    def __call__(self, a, axis=None, keepdims=False):
        self.variables = (a,)

        if axis is not None and not hasattr(axis, "__iter__"):
            axis = (axis,)
        if axis is None:
            axis = tuple(i for i in range(len(a.shape)))
        self.axis = axis
        self.size = np.prod([a.shape[i] for i in self.axis])
        self.keepdims = keepdims
        out = a.data.mean(axis=axis, keepdims=keepdims)
        out = np.array(out)
        self.outshape = out.shape
        return out

    def backward(self, grad, **kwargs):
        a = self.variables[0]
        # print(grad.shape, a.shape, self.axis, self.outshape)
        # print(grad.sum() / self.size)
        grad = self.broadcastable(grad, self.outshape)
        # grad = self.broadcastable(grad, a.shape)
        # print(grad.shape, a.shape, self.axis, self.outshape)
        if not self.keepdims:
            index = [slice(None) for i in range(a.ndim)]
            for i in self.axis:
                index[i] = np.newaxis
            grad = grad[tuple(index)]
        grad = np.broadcast_to(grad, a.data.shape).astype(float)
        # print("mean",grad.shape, a.shape, self.axis, self.outshape, grad[0])
        self.variables[0].backward(grad / self.size)


class Variance(Operation):
    testLevel = 2
    def __call__(self, a, axis=None, keepdims=False, ddof=0):
        self.variables = (a,)
        if axis is not None and not hasattr(axis, "__iter__"):
            axis = (axis,)
        if axis is None:
            axis = tuple(i for i in range(len(a.shape)))
        self.axis = axis
        self.size = np.prod([a.shape[i] for i in self.axis])
        self.keepdims = keepdims
        self.ddof = ddof
        out = np.var(a.data, axis=axis, keepdims=keepdims, ddof=ddof)
        out = np.array(out)
        self.outshape = out.shape
        return out


    def backward(self, grad, **kwargs):
        a = self.variables[0]
        N = self.size - self.ddof
        grad = self.broadcastable(grad, self.outshape)
        if not self.keepdims:
            index = [slice(None)] * a.ndim
            for i in self.axis:
                index[i] = np.newaxis
            grad = grad[tuple(index)]
        back = (2.0 / N) * (a.data - a.data.mean(axis=self.axis, keepdims=True))
        self.variables[0].backward(back * grad)


class StdDev(Operation):
    def _grad_preprocess(self, grad: Any) -> np.ndarray:
        a = self.variables[0]
        return np.asarray(grad) / (2 * np.sqrt(a.data.var(**self.kwargs)))

    def __call__(self, a, axis=None, keepdims=False, ddof=0):
        self.variables = (a,)

        if axis is not None and not hasattr(axis, "__iter__"):
            axis = (axis,)

        self.kwargs = dict(axis=axis, keepdims=keepdims, ddof=ddof)
        return getattr(a.data, 'std')(**self.kwargs)


    def backward(self, grad, **kwargs):
        a = self.variables[0]
        if isinstance(self.kwargs["axis"], Sequence) and len(self.kwargs["axis"]) == 0:
            self.variables[0].backward(np.zeros(a.shape, dtype=float))
            return

        N = (
            a.size
            if self.kwargs["axis"] is None
            else np.prod([a.shape[i] for i in self.kwargs["axis"]])
        )
        N -= self.kwargs["ddof"]

        grad = self._grad_preprocess(grad)
        if grad.ndim == 0:
            grad = np.full(a.shape, grad, dtype=float)
        else:
            if not self.kwargs["keepdims"]:
                index = [slice(None)] * a.ndim
                for i in self.kwargs["axis"]:
                    index[i] = np.newaxis
                grad = grad[tuple(index)]
        back = (2.0 / N) * (
            a.data - a.data.mean(axis=self.kwargs["axis"], keepdims=True)
        )
        self.variables[0].backward(back * grad)
