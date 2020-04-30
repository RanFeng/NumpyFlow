import nf
import nf.nn.functional as F
import numpy as np
import torch
from torch.nn.functional import conv2d, max_pool2d
from torch.autograd import Variable
from nf.nn.modules.batchnorm import BatchNorm2d as nfBN2d
from torch.nn.modules.batchnorm import BatchNorm2d
from time import time
def test_conv2d(k, d, s=1):
    x = np.random.random([1,1,28,28]).astype(np.float32)
    k = np.random.random([1,1, 3, 3]).astype(np.float32)
    # p = (1,1,1,1)
    p = (0,0,0,0)
    dilation = 1
    stride = 1
    # x = np.ones([1, 1, 64, 48])
    # k = np.ones([1, 1, 3, 3])
    def nf_conv2d(x, k):
        x = nf.Tensor(x,requires_grad=True)
        k = nf.Tensor(k,requires_grad=True)
        t1 = time()
        out = F.conv2d(x,k, stride=stride, dilation=dilation, padding=p)
        out = F.max_pool2d(out, stride=2, pool_size=2)
        # out = F.conv2d(out,k, stride=stride, dilation=dilation, padding=p)
        # print("nfr:",time()-t1)
        out.backward()
        # print(k.grad)
        return [out.numpy(), x.grad, k.grad]

    def th_conv2d(x, k):
        k = Variable(torch.from_numpy(k),requires_grad=True)
        x = Variable(torch.from_numpy(x),requires_grad=True)
        t1 = time()
        out = conv2d(x, k, None, stride=stride, padding=(p[0],p[2]), dilation=dilation)
        out = max_pool2d(out, kernel_size=2, stride=2)
        # out = conv2d(out, k, None, stride=stride, padding=(p[0],p[2]), dilation=dilation)
        # print("tfr:",time() - t1)
        out.backward(torch.ones_like(out))
        return [out, x.grad, k.grad]

    thr = th_conv2d(x, k)
    nfr = nf_conv2d(x, k)
    # print(nfr[0].shape)
    # nfr = [0,0,0]


    for ni, ki in zip(nfr, thr):
        # ni = ni.numpy()
        # ki = ki.detach().numpy()
        try:
            ni = ni
            ki = ki.detach().numpy()
        except:
            print("出错")
            continue
        # print(ni.shape)
        # print(ki.shape)
        try:
            print(np.allclose(ni, ki))
            if not np.allclose(ni, ki):
                print(ni)
                print(ki)
        except:
            print("不合适")
            pass

def test_pool2d(k, d, s=1):
    # x = [[
    #     [[0,1,0,0],
    #      [0,0,0,0],
    #      [1,0,0,1],
    #      [0,1,1,0],],
    #     [[0,0,0,0],[0,0,0,0],
    #     [0,0,0,0],[0,0,0,0],],
    #     [[0,0,0,0],[0,0,0,0],
    #     [0,0,0,0],[0,0,0,0],],
    #     [[0,0,0,0],[0,0,0,0],
    #     [0,0,0,0],[0,0,0,0],]
    # ]]
    # x = np.array(x) * 1.0
    # print(x.shape)
    x = np.random.random([40, 40, 64, 64])
    k = np.random.random([1, 1, 3, 3])
    w = np.random.random([4*3*3, 64])
    pool_size = 2
    p = (0,0,0,0)
    stride = 2
    # x = np.ones([1, 1, 64, 48])
    # k = np.ones([1, 1, 3, 3])
    def nf_conv2d(x, w, k):
        x = nf.Tensor(x,requires_grad=True)
        w = nf.Tensor(w,requires_grad=True)

        t1 = time()
        # x = F.conv2d(x, k, None, 'same', 1)
        out = F.max_pool2d(x, stride=stride, pool_size=pool_size)
        # out = out.reshape([-1,4*3*3])
        # out = out @ w
        # out = F.conv2d(out,k, stride=stride, dilation=dilation, padding=p)
        out.backward()
        print("nfr:",time()-t1)

        # print(k.grad)
        return [out.numpy(), x.grad]

    def th_conv2d(x, w, k):
        x = Variable(torch.from_numpy(x),requires_grad=True)
        w = Variable(torch.from_numpy(w),requires_grad=True)
        t1 = time()
        out = max_pool2d(x, kernel_size=pool_size, stride=stride)
        # out = out.reshape([-1,4*3*3])
        # out = out @ w

        # out = conv2d(out, k, None, stride=stride, padding=(p[0],p[2]), dilation=dilation)
        out.backward(torch.ones_like(out))
        print("tfr:",time() - t1)

        return [out, x.grad]

    thr = th_conv2d(x, w, k)
    nfr = nf_conv2d(x, w, k)
    # print(nfr[0].shape)
    # nfr = [0,0,0]


    for ni, ki in zip(nfr, thr):
        # ni = ni.numpy()
        # ki = ki.detach().numpy()
        try:
            ni = ni
            ki = ki.detach().numpy()
        except:
            print("出错")
            continue
        print(ni.shape)
        print(ki.shape)
        try:
            a = np.allclose(ni, ki)
            print(a)
            if not a:
                print(ni)
                print(ki)
        except:
            print("不合适")
            pass

def test_bn2d(k, d, s=1):
    x = np.random.random([2, 2, 2, 2]).astype(np.float32)
    feas = x.shape[1]
    # x = np.ones([1, 1, 64, 48])
    # k = np.ones([1, 1, 3, 3])
    def nf_conv2d(x, k):
        x = nf.Tensor(x, requires_grad=True)
        t1 = time()
        f1 = nfBN2d(feas)
        out = f1(x)
        out.backward()
        print("nfr:",time()-t1)
        # print(out)
        # print(k.grad)
        return [out.numpy(), x.grad]

    def th_conv2d(x, k):
        x = Variable(torch.from_numpy(x),requires_grad=True)
        t1 = time()
        f1 = BatchNorm2d(feas)
        f1.train()
        out = f1(x)
        out.backward(torch.ones_like(out))
        print("tfr:",time() - t1)

        return [out, x.grad]

    thr = th_conv2d(x, k)
    nfr = nf_conv2d(x, k)
    # print(nfr[0].shape)
    # nfr = [0,0,0]


    for ni, ki in zip(nfr, thr):
        # ni = ni.numpy()
        # ki = ki.detach().numpy()
        try:
            ni = ni
            ki = ki.detach().numpy()
        except:
            print("出错")
            continue
        # print(ni.shape)
        # print(ki.shape)
        try:
            a = np.allclose(ni, ki)
            print(a)
            if not a:
                print(ni)
                print(ki)
        except:
            print("不合适")
            pass


def test_mean(k, d, s=1):
    x = np.random.random([3, 2, 4, 4]).astype(np.float64)
    feas = x.shape[1]
    axis = (0,2,3)
    keepdims = False
    eps = 1.e-5
    print(np.var(x, axis=axis, keepdims=keepdims, ddof=1))
    # x = np.ones([1, 1, 64, 48])
    # k = np.ones([1, 1, 3, 3])
    def nf_conv2d(x, k):
        x = nf.Tensor(x, requires_grad=True)
        t1 = time()
        running_mean = nf.mean(x, axis=axis, keepdims=True)
        running_var = nf.var(x, ddof=1, axis=axis, keepdims=True)
        out = (x - running_mean)
        out = out / (running_var + eps) ** 0.5
        # out = x / running_var
        # out = nf.var(x, ddof=1, axis=axis, keepdims=keepdims)
        out.backward()
        print("nfr:",time()-t1)
        # print(out)
        # print(k.grad)
        return [out.numpy(), x.grad]

    def th_conv2d(x, k):
        x = Variable(torch.from_numpy(x),requires_grad=True)
        t1 = time()
        running_mean = torch.mean(x, axis=axis, keepdims=True)
        running_var = torch.var(x, axis=axis, keepdims=True)
        out = (x - running_mean)
        out = out / (running_var+eps) ** 0.5
        # out = x / running_var
        # out = torch.var(x, dim=axis, keepdim=keepdims)
        out.backward(torch.ones_like(out))
        print("tfr:",time() - t1)

        return [out, x.grad]

    thr = th_conv2d(x, k)
    nfr = nf_conv2d(x, k)
    # print(nfr[0].shape)
    # nfr = [0,0,0]


    for ni, ki in zip(nfr, thr):
        # ni = ni.numpy()
        # ki = ki.detach().numpy()
        try:
            ni = ni
            ki = ki.detach().numpy()
        except:
            print("出错")
            continue
        # print(ni.shape)
        # print(ki.shape)
        try:
            a = np.allclose(ni, ki)
            print(a)
            if not a:
                print(ni)
                print(ki)
        except:
            print("不合适")
            pass



if __name__ == '__main__':
    # np.random.seed(20)
    # for s in range(1,7):
    #     for d in range(1,10):
    #         for k in range(1,14):
    #             test_conv2d((k,k), d, s)
    # test_conv2d((3,3), 2, 3)
    # for s in range(1,7):
    # for k in range(2,14):
    #     test_pool2d(k,1,k)
    # test_mean(5,1,5)
    # test_conv2d(5,1,5)
    test_pool2d(5,1,5)























