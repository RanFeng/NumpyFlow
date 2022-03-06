import nf
from nf import Tensor
import nf.nn.modules as nn
import nf.nn.functional as F
import numpy as np
from nf.optimizer.sgd import SGD
import mnist
from time import time



def categorical_crossentropy(output, target, from_logits=False):
    output /= output.sum(axis=-1, keepdims=True)
    # output = np.clip(output, 1e-7, 1 - 1e-7)
    a = target * -nf.log(output)
    return nf.sum(a, axis=-1, keepdims=False)

class NfModel(nn.Module):
    def __init__(self):
        super(NfModel, self).__init__()

        self.c1 = nn.Conv2d(1, 32, 3, stride=1, padding='same')
        self.c2 = nn.Conv2d(32, 64, 3, stride=1, padding='same')
        self.c3 = nn.Conv2d(64, 4, 3, stride=1, padding='same')
        self.b1 = nn.BatchNorm2d(3)
        self.b2 = nn.BatchNorm2d(16)
        self.b3 = nn.BatchNorm2d(4)
        # self.fc1 = nn.Linear(1*28*28//4, CLASS_SIZE)
        self.fc1 = nn.Linear(4 * 3 * 3, 64)
        # self.fc2 = nn.Linear(64, 64)
        self.fc2 = nn.Linear(64, CLASS_SIZE)


    def forward(self, x):
        if isinstance(x, np.ndarray):
            x = Tensor(x)
        # x = F.relu(self.c1(x))
        # print("x1.shape", x.shape)
        x = self.c1(x)
        x = F.max_pool2d(x)
        x = F.relu(self.c2(x))
        x = F.max_pool2d(x)
        x = F.relu(self.c3(x))
        x = F.max_pool2d(x)
        # print("XXX", x.shape)
        x = x.reshape([-1, 4*3*3])
        # print("XXX", x.shape)
        x = F.relu(self.fc1(x))
        x = F.softmax(self.fc2(x), -1)
        # x = F.softmax(self.fc1(x), -1)
        # x = self.fc1(x)
        return x




if __name__ =='__main__':
    BATCH_SIZE = 120
    IMAGE_SIZE = 28
    CLASS_SIZE = 10
    x_train, y_train, x_test, y_test = mnist.load()
    x_train = x_train.astype(float).reshape([-1,1,28,28])
    x_test = x_test.astype(float).reshape([-1,1,28,28])
    x_train = x_train / x_train.max()
    x_test  = x_test / x_test.max()
    y_train = np.eye(10)[y_train]
    y_test = np.eye(10)[y_test]

    print(x_train.shape, y_train.shape)
    print(x_test.shape, y_test.shape)
    print(x_train.max(), x_test.max())

    nfnet = NfModel()
    nfopt = SGD(nfnet.parameters(), lr=1e-4, momentum=0.9,nesterov=True)

    # batch = Tensor(x_test[0:BATCH_SIZE])
    # gt = Tensor(y_test[0:BATCH_SIZE])
    # nfr = nfnet(batch).numpy()
    # nfr = np.argmax(nfr, axis=-1)
    # gt = np.argmax(gt.numpy(), axis=-1)
    # print(nfr)
    # print(gt)
    # acc = (nfr == gt).sum()
    # print("初始化正确率",acc / nfr.shape[0])
    epch = 3
    circle = 500
    for j in range(epch):
        # print()
        t1 = time()
        for i in range(circle):
            t2 = time()
            batch = Tensor(x_train[i*BATCH_SIZE:(i+1)*BATCH_SIZE], requires_grad=True)
            gt = Tensor(y_train[i*BATCH_SIZE:(i+1)*BATCH_SIZE])
            nfr = nfnet(batch)
            loss = categorical_crossentropy(nfr, gt)
            nfopt.zero_grad()
            loss.backward()
            nfopt.step()
            print(i,time()-t2)
        # if((j+1) % 5 == 0):
        batch = Tensor(x_test)
        gt = Tensor(y_test)
        nfr = nfnet(batch).numpy()
        nfr = np.argmax(nfr, axis=-1)
        gt = np.argmax(gt.numpy(), axis=-1)
        # print(nfr)
        # print(gt.shape)
        acc = (nfr == gt).sum()
        print("epoch:",j," ", time()-t1," ", acc / nfr.shape[0],' ', loss.mean().numpy())
    # print(thr)
    # print(nfr)



