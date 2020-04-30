from time import time
import numpy as np
cimport numpy as np
cimport cython
from cython.parallel import prange

ctypedef np.float32_t FLOAT32
ctypedef np.int32_t INT32
ctypedef np.long_t LONG
ctypedef np.float64_t FLOAT64
ctypedef fused FLOAT:
    FLOAT32
    FLOAT64

def test():
    print("aaaaaaaaaa啊啊")

@cython.boundscheck(False)  # 数组确定不会越界？
@cython.wraparound(False)   # 确定不会使用负号作为数组index？
def maxpool2d_fw(np.ndarray[FLOAT, ndim=6] xview):
    cdef long B = xview.shape[0]
    cdef long C = xview.shape[1]
    cdef long H = xview.shape[2]
    cdef long W = xview.shape[3]
    cdef long K = xview.shape[4]
    cdef long L = xview.shape[5]
    cdef long b,c,h,w,k,l
    cdef FLOAT r
    cdef np.ndarray[FLOAT, ndim=4] result = np.zeros((B,C,H,W), dtype=xview.dtype)
    cdef FLOAT[:,:,:,::1] result_view = result
    cdef FLOAT[:,:,:,:,:,:] xview_view = xview
    for b in prange(B, nogil=True, schedule='guided'):
        for c in range(C):
            for h in range(H):
                for w in range(W):
                    r = xview[b,c,h,w,0,0]
                    for k in range(K):
                        for l in range(L):
                            r = r if r > xview_view[b,c,h,w,k,l] else xview_view[b,c,h,w,k,l]
                    result_view[b,c,h,w] = r
    return result

@cython.boundscheck(False)  # 数组确定不会越界？
@cython.wraparound(False)   # 确定不会使用负号作为数组index？
def maxpool2d_bp(np.ndarray[FLOAT, ndim=6] xview, np.ndarray[FLOAT, ndim=6] dst, np.ndarray[FLOAT, ndim=4] grad):
    cdef long B = xview.shape[0]
    cdef long C = xview.shape[1]
    cdef long H = xview.shape[2]
    cdef long W = xview.shape[3]
    cdef long K = xview.shape[4]
    cdef long L = xview.shape[5]
    cdef long b,c,h,w,k,l,k0,k1
    cdef FLOAT[:,:,:,:] grad_view = grad
    cdef FLOAT[:,:,:,:,:,:] dst_view = dst
    cdef FLOAT[:,:,:,:,:,:] xview_view = xview
    for b in prange(B, nogil=True, schedule='guided'):
        for c in range(C):
            for h in range(H):
                for w in range(W):
                    k0 = 0
                    k1 = 0
                    for k in range(K):
                        for l in range(L):
                            if xview_view[b,c,h,w,k0,k1] < xview_view[b,c,h,w,k,l]:
                                k0 = k
                                k1 = l
                    dst_view[b,c,h,w,k0,k1] += grad_view[b,c,h,w]

# def relu(np.ndarray[FLOAT] x):
#     """
#     :param x: relu函数改变x的值，逐位修改x
#     :return:
#     """