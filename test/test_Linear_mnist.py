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
        self.fc1 = nn.Linear(IMAGE_SIZE * IMAGE_SIZE, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, CLASS_SIZE)

    def forward(self, x):
        if isinstance(x, np.ndarray):
            x = Tensor(x)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.softmax(self.fc3(x), -1)
        return x




if __name__ =='__main__':
    BATCH_SIZE = 60
    IMAGE_SIZE = 28
    CLASS_SIZE = 10
    x_train, y_train, x_test, y_test = mnist.load()
    x_train = x_train.astype(np.float)
    x_test = x_test.astype(np.float)
    x_train = x_train / x_train.max()
    x_test  = x_test / x_test.max()
    y_train = np.eye(10)[y_train]
    y_test = np.eye(10)[y_test]
    _, input_size = x_train.shape
    _, output_size = y_train.shape
    print(x_train.shape, y_train.shape)
    print(x_test.shape, y_test.shape)
    print(x_train.max(), x_test.max())

    nfnet = NfModel()
    nfopt = SGD(nfnet.parameters(), lr=1e-4, momentum=0.9,nesterov=True)

    batch = Tensor(x_test)
    gt = Tensor(y_test)
    nfr = nfnet(batch).numpy()
    nfr = np.argmax(nfr, axis=-1)
    gt = np.argmax(gt.numpy(), axis=-1)
    # print(nfr)
    # print(gt)
    acc = (nfr == gt).sum()
    print(acc / nfr.shape[0])
    epch = 30
    circle = 999
    for j in range(epch):
        # print()
        t1 = time()
        for i in range(circle):
            batch = Tensor(x_train[i*BATCH_SIZE:(i+1)*BATCH_SIZE])
            gt = Tensor(y_train[i*BATCH_SIZE:(i+1)*BATCH_SIZE])
            nfr = nfnet(batch)
            loss = categorical_crossentropy(nfr, gt)
            nfopt.zero_grad()
            loss.backward()
            nfopt.step()
        # print(time()-t1)
        # if((j+1) % 5 == 0):
        batch = Tensor(x_test)
        gt = Tensor(y_test)
        nfr = nfnet(batch).numpy()
        nfr = np.argmax(nfr, axis=-1)
        gt = np.argmax(gt.numpy(), axis=-1)
        # print(nfr)
        # print(gt)
        acc = (nfr == gt).sum()
        print("epoch: ",j, time()-t1,"\t", acc / nfr.shape[0],'\t', loss.mean().numpy())
    # print(thr)
    # print(nfr)



