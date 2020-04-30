# NumpyFlow

详细文档请参照：https://blog.csdn.net/kid_14_12/article/details/105852626



待添加.....



# Tensor

封装numpy数组，是NumpyFlow的数据载体，相当于torch中的Tensor。



# 已完成

## 基本类
- [x] Operation：用于支持基本的运算及对应的梯度计算，是支持自动微分的基本算子
- [x] Tensor：
- [x] Optimizer
- [x] Module

## Operation
- [x] Assign、Add、Multiply、Subtract
- [x] Divide、Negative、Positive、Power
- [x] Exp、Log、Log2、Log10
- [x] MatMul、EinSum


## Optimizer
- [x] sgd
- [ ] adam


## 初始化方法
- [x] Kaiming初始化
- [ ] Xavier初始化
- [ ] 随机初始化
- [x] 填充0初始化
- [x] 填充1初始化

## Module

### 核心网络层

- [x] Linear
- [x] Relu、Sigmid、Softmax
- [ ] Conv1D
- [x] Conv2D
- [ ] Conv3D
- [ ] MaxPool1D
- [x] MaxPool2D
- [ ] MaxPool3D
- [ ] BatchNorm1D
- [x] BatchNorm2D
- [ ] BatchNorm3D
- [ ] Reshape
- [ ] Permute
- [ ] Flatten
- [ ] RepeatVector
- [ ] Lambda
- [ ] ActivityRegularization
- [ ] Masking
- [ ] SpatialDropout1D
- [ ] SpatialDropout2D
- [ ] SpatialDropout3D


### 局部连接层和循环层

- [ ] LocallyConnected1D
- [ ] LocallyConnected2D
- [ ] RNN
- [ ] GRU
- [ ] LSTM
- [ ] ConvLSTM2D
- [ ] SimpleRNNCell
- [ ] GRUCell
- [ ] LSTMCell
- [ ] CuDNNGRU
- [ ] CuDNNLSTM
- [ ] Embedding























