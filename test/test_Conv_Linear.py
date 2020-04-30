import nf
from nf import Tensor, Parameter
import nf.Module as nm
from nf.Module import Functional as nmF
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as nnF
from torch.autograd import *
from nf.Optimizer.sgd import SGD
from torch import optim

# def categorical_crossentropy(output, target, from_logits=False):
#     output /= output.sum(axis=-1, keepdims=True)
#     # output = np.clip(output, 1e-7, 1 - 1e-7)
#     a = target * -np.log(output)
#     return np.sum(, axis=-1, keepdims=False)

def netBuild(bd):
    f1 = bd.Linear(2,3)
    # y = backend.ReLU()
    f2 = bd.Linear(3,3)
    f3 = bd.Softmax(1)
    f4 = bd.Sigmoid()
    f5 = bd.Sigmoid()
    # f1 = f3
    # f2 = f4

    # return lambda z: f3(f4(f2(f1(z))))
    return lambda z: f1(z)


class ThModel(nn.Module):
    def __init__(self):
        super(ThModel, self).__init__()

        self.fc1 = nn.Linear(3, 2)
        # self.fc2 = nn.Linear(2, 4)
        # self.fc3 = nn.Linear(4, 5)

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
        # self.fc2 = nm.Linear(2, 4)
        # self.fc3 = nm.Linear(4, 5)

    def forward(self, x):
        if isinstance(x, np.ndarray):
            x = Tensor(x)

        x = nmF.relu(self.fc1(x))
        # x = nmF.relu(self.fc2(x))
        # x = nmF.softmax(self.fc3(x), 1)
        return x


def setup_seed(seed):
    torch.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    # random.seed(seed)
    # torch.backends.cudnn.deterministic = True

def th_layer(z):
    z = Variable(torch.from_numpy(z), requires_grad=True)
    f1 = nn.Linear(2,3)

    net = netBuild(nn)
    for name, parameters in f1.named_parameters():
        print(name, parameters)
    print(f1.state_dict())
    # print(f1.__name__())
    r = f1(z)
    # print(r.sum(axis=-1, keepdims=True))
    # r = r / r.sum(axis=-1, keepdims=True)
    # a = torch.ones_like(r) * -torch.log(r)
    # r = a.sum(axis=-1, keepdims=False)
    r.backward(torch.ones_like(r))
    return [r.detach().numpy(), z.grad.numpy()]

def nf_layer(z):
    z = Tensor(z, requires_grad=True)
    net = netBuild(nm)
    r = net(z)
    f1 = nm.Linear(2, 3)
    r = f1(z)
    # print(r.sum(axis=-1, keepdims=True))
    # r = r / nf.sum(r, axis=-1, keepdims=True)
    # a = nf.ones_like(r) * -nf.log(r)
    # r = a.sum(axis=-1, keepdims=False)
    r.backward()
    return [r.data, z.grad]



if __name__ =='__main__':
    # setup_seed(20)
    #
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



