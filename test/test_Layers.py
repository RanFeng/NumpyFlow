import nf
from nf import Tensor, Parameter
import nf.nn.modules as nm
from nf.nn import functional as nmF
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as nnF
from torch.autograd import *
from nf.optimizer.sgd import SGD
from torch import optim

# def categorical_crossentropy(output, target, from_logits=False):
#     output /= output.sum(axis=-1, keepdims=True)
#     # output = np.clip(output, 1e-7, 1 - 1e-7)
#     a = target * -np.log(output)
#     return np.sum(, axis=-1, keepdims=False)

def test_Linear():
    class ThModel(nn.Module):
        def __init__(self):
            super(ThModel, self).__init__()

            self.fc1 = nn.Linear(3, 2)
            self.fc2 = nn.Linear(2, 4)
            self.fc3 = nn.Linear(4, 5)

        def forward(self, x):
            if isinstance(x, np.ndarray):
                x = torch.from_numpy(x)
            # x = self.pool(nnF.relu(self.conv1(x)))
            # x = self.pool(nnF.relu(self.conv2(x)))
            # x = x.view(-1, 16 * 5 * 5)
            x = nnF.relu(self.fc1(x))
            # x = nnF.relu(self.fc2(x))
            # x = nnF.softmax(self.fc3(x), 1)
            return x

    class NfModel(nm.Module):
        def __init__(self):
            super(NfModel, self).__init__()

            self.fc1 = nm.Linear(3, 2)
            self.fc2 = nm.Linear(2, 4)
            self.fc3 = nm.Linear(4, 5)

        def forward(self, x):
            if isinstance(x, np.ndarray):
                x = Tensor(x)

            x = nmF.relu(self.fc1(x))
            # x = nmF.relu(self.fc2(x))
            # x = nmF.softmax(self.fc3(x), 1)
            return x

    z = np.random.random([5,3]).astype(np.float32) * 20

    thnet = ThModel()
    thp = thnet.state_dict()
    for k in thp.keys():
        thp[k] = Tensor(thp[k].numpy())

    nfnet = NfModel()
    nfnet.load_state_dict(thp)
    thopt = optim.SGD(thnet.parameters(), lr=1e-3, momentum=0.4,nesterov=True)
    nfopt = SGD(nfnet.parameters(), lr=1e-3, momentum=0.4,nesterov=True)

    circle = 800

    for i in range(circle):
        thr = thnet(torch.from_numpy(z))
        loss = (3.-thr)
        thopt.zero_grad()
        loss.backward(torch.ones_like(loss))
        thopt.step()

    for i in range(circle):
        nfr = nfnet(z)
        loss = (3.-nfr)
        nfopt.zero_grad()
        loss.backward()
        nfopt.step()

    thr = thnet(z).detach().numpy()
    nfr = nfnet(z).numpy()
    print(thr)
    print(nfr)

def test_Conv():
    class ThModel(nn.Module):
        def __init__(self):
            super(ThModel, self).__init__()

            self.c1 = nn.Conv2d(1, 3, 3, stride=2)
            self.c2 = nn.Conv2d(3, 9, (3, 5), stride=(2, 1), padding=(4, 2))
            self.c3 = nn.Conv2d(9, 1, (3, 5), stride=(2, 1), padding=(4, 2))
            self.b1 = nn.BatchNorm2d(3)
            self.b2 = nn.BatchNorm2d(9)
            self.b3 = nn.BatchNorm2d(1)

        def forward(self, x):
            if isinstance(x, np.ndarray):
                x = torch.from_numpy(x)

            x = nnF.relu(self.b1(self.c1(x)))
            x = nnF.relu(self.b2(self.c2(x)))
            x = nnF.relu(self.b3(self.c3(x)))
            return x

    class NfModel(nm.Module):
        def __init__(self):
            super(NfModel, self).__init__()

            self.c1 = nm.Conv2d(1, 3, 3, stride=2)
            self.c2 = nm.Conv2d(3, 9, (3, 5), stride=(2, 1), padding=(4,4,2,2))
            self.c3 = nm.Conv2d(9, 1, (3, 5), stride=(2, 1), padding=(4,4,2,2))
            self.b1 = nm.BatchNorm2d(3)
            self.b2 = nm.BatchNorm2d(9)
            self.b3 = nm.BatchNorm2d(1)

        def forward(self, x):
            if isinstance(x, np.ndarray):
                x = Tensor(x)

            x = nmF.relu(self.b1(self.c1(x)))
            x = nmF.relu(self.b2(self.c2(x)))
            x = nmF.relu(self.b3(self.c3(x)))
            return x

    z = np.random.random([4,1,7,7]).astype(np.float32)

    thnet = ThModel()
    thp = thnet.state_dict()
    for k in thp.keys():
        thp[k] = Tensor(thp[k].numpy())

    nfnet = NfModel()
    nfnet.load_state_dict(thp)
    thopt = optim.SGD(thnet.parameters(), lr=1e-3, momentum=0.4,nesterov=True)
    nfopt = SGD(nfnet.parameters(), lr=1e-3, momentum=0.4,nesterov=True)

    # l = list(nfnet.parameters())
    # print(l)

    circle = 10

    for i in range(circle):
        thr = thnet(z)
        loss = (3.-thr)
        thopt.zero_grad()
        loss.backward(torch.ones_like(loss))
        # print(thnet.c1.weight.grad)
        thopt.step()

    for i in range(circle):
        nfr = nfnet(z)
        loss = (3.-nfr)
        nfopt.zero_grad()
        loss.backward()
        # print(nfnet.c1.weight.grad)
        nfopt.step()

    thr = thnet(z).detach().numpy()
    nfr = nfnet(z).numpy()
    print(thr)
    print(nfr)
    try:
        a = np.allclose(thr, nfr)
        print(a)
        # if not a:
        #     print(thr)
        #     print(nfr)
    except:
        print("不合适")
        pass



def setup_seed(seed):
    torch.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    # random.seed(seed)
    # torch.backends.cudnn.deterministic = True



if __name__ =='__main__':
    # setup_seed(20)
    test_Conv()
    # test_Linear()
    # print(thr/nfr)
    # for t,n in zip(thr, nfr):
    #     print(t)
    #     print(n)
        # print(np.allclose(t, n, rtol=1.e-5,atol=1.e-8))

    # print(thr[1])
    # print(nfr[1])



    # model = TheModelClass()
    # print("Model's state_dict:")
    # # print(model.state_dict())
    # a = model.state_dict()
    # model.load_state_dict(a)
    # print(a)

    # Print optimizer's state_dict
    # print("Optimizer's state_dict:")
    # for var_name in optimizer.state_dict():
    #     print(var_name, "\t", optimizer.state_dict()[var_name])



