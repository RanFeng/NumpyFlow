from nf import Tensor
import numpy as np
import torch
from torch.autograd import *
from time import time


def func(x,y,z):
    # f0 = (x[1,0].T * y[0,1].T).T * z * x
    # f1 = f0 * (x + y + z) * y * y * y * (y+z) #! 有错[9,23,29]
    # f2 = y[0,3] + x[0,2]
    # f3 = y * y - z
    # f4 = z - x
    # f5 = -x.flatten() + y.flatten() - (x*z).flatten() * 2.0
    # f6 = f1[1,3] + f1[0,3] * f2 - z[0,1] ** 2.2
    # f7 = f3 + f4 + f6
    # f8 = f7 - f3 + f4 * 3.6
    # f9 = f8.flatten() / f5 + f7.flatten()
    # f10 = -f9 * f5
    # f11 = ((x*z) @ x.transpose(3, 4) @ y.permute(0,4,2,3,1)).transpose(0,4)
    # f12 = f11.transpose(3,4).flatten() * 5.0 ** x.transpose(1,4).flatten() / y.flatten() * (x/z).flatten() + 2.0
    # f13 = f10.reshape(f11.shape) * f11 / f12.reshape(f11.shape)
    # f14 = (x.transpose(3,4) @ y).permute(0,2,4,3,1) @ f13.permute(4,2,0,1,3)
    # f15 = f14.sum() * f14.mean((0,2))
    f15 = x @ y @ z
    return f15

def th_grad_Test(x,y,z):
    x = Variable(torch.from_numpy(x), requires_grad=True)
    y = Variable(torch.from_numpy(y), requires_grad=True)
    z = Variable(torch.from_numpy(z), requires_grad=True)
    def hook(grad, v=None):
        print(v, int(grad))
    t1 = time()
    f9 = func(x, y, z)
    t2 = time() - t1
    t1 = time()
    f9.backward(torch.ones_like(f9), retain_graph=True)
    print("th", t2,time() - t1)

    return [x.grad.numpy(), y.grad.numpy(), z.grad.numpy()]


def nf_grad_Test(x,y,z):
    x = Tensor(x, requires_grad=True)
    y = Tensor(y, requires_grad=True)
    z = Tensor(z, requires_grad=True)
    t1 = time()
    f9 = func(x,y,z)
    t2 = time() - t1
    t1 = time()
    f9.backward()
    print("nf", t2,time() - t1)
    return [x.grad, y.grad, z.grad]


def test1():
    np.random.seed(28)
    x = np.random.random([2,4,6,3,4])
    y = np.random.random([2,4,6,4,7])
    z = np.random.random([2,4,1,7,1])

    grad_th = th_grad_Test(x,y,z)
    grad_nf = nf_grad_Test(x,y,z)

    for (thi, nfi) in zip(grad_th, grad_nf):
        a = np.allclose(nfi, thi)
        print(a)


if __name__ =='__main__':
    test1()
    # np.random.seed(28)
    # x = np.random.random([3,4])
    # y = np.random.random([3,4])

    # x,y,z = 5,2,3
    #
    # x = Tensor(x, requires_grad=True)
    # y = Tensor(y, requires_grad=True)
    # z = Tensor(z, requires_grad=True)
    #
    # f1 = y+z
    # f2 = y*f1
    # f3 = y*f2
    # f4 = y*f3
    # f5 = z*f4
    # f6 = x*f5
    #
    # print(f1,f2,f3,f4,f5,f6)
    #
    # f6.backward()
    #
    # print(x.grad,y.grad,z.grad)
    # print(f1.grad,f2.grad,f3.grad)
    # print(f4.grad,f5.grad,f6.grad)
    # print(id(x.grad), id(y.grad), id(z.grad))
    # z.backward()


