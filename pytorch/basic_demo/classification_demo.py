"""
基础的分类任务
"""
import pickle
import torch
import numpy as np
import torch.nn.functional as func
from torch import nn
from torch import optim
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

with open("./data/mnist.pkl", "rb") as f:
    ((x_train, y_train), (x_valid, y_valid), _) = pickle.load(f, encoding="latin-1")
# 打印下数据格式
print(x_train.shape)

# 将数据转换成tensor的格式
x_train, y_train, x_valid, y_valid = map(torch.tensor, (x_train, y_train, x_valid, y_valid))

# 打印一下转换后的数据
print(x_train, y_train)
print(x_train.shape)
print(y_train.min(), y_train.max())

# 做个简单的案例就用torch.nn.functional就行了,实际做模型的参数的训练还是要用torch.nn.Module
# 分类任务用交叉熵损失函数
loss_func = func.cross_entropy
# 取一个小batch
bs = 64
xb = x_train[0:bs]
yb = y_train[0:bs]
# 随机一个权重参数
weights = torch.randn([784, 10], dtype=torch.float, requires_grad=True)
bs = 64
bias = torch.zeros(10, requires_grad=True)


# 一个简单的模型案例
def model(input_data):
    return input_data.mm(weights) + bias


# 打印损失值
print('\nloss:', end='')
print(loss_func(model(xb), yb))


# 使用nn.Module来做一个模型
# 直接继承nn.Module并调用父类的构造,nn.Module可以自己进行反向传播
# 可学习的权重参数通过named_parameters()或者parameters()返回一个迭代器
class MnistNn(nn.Module):
    def __init__(self):
        super().__init__()
        # 线性模块,Applies a linear transformation to the incoming data: :math:`y = xA^T + b
        self.hidden1 = nn.Linear(784, 128)
        self.hidden2 = nn.Linear(128, 256)
        self.out = nn.Linear(256, 10)

    def forward(self, x):
        # 经过第一个隐层,用relu激活一下
        x = func.relu(self.hidden1(x))
        # 加上dropout
        x = func.dropout(x, p=0.5)
        x = func.relu(self.hidden2(x))
        x = self.out(x)
        return x


# 实例化一个网络模型,并打印看看
net = MnistNn()
# print(net)
# # 模型里的参数也能打印出来看看
# for name, parameter in net.named_parameters():
#     print(name, parameter, parameter.size())

# 使用专门的的类来加载数据,TensorDataset和DataLoader
train_ds = TensorDataset(x_train, y_train)
# 按批次加载,并且做洗牌操作
train_dl = DataLoader(train_ds, batch_size=bs, shuffle=True)

valid_ds = TensorDataset(x_valid, y_valid)
valid_dl = DataLoader(valid_ds, batch_size=bs * 2)


# 训练的时候一般加上model.train(),这样能正常使用Batch Normalization和 Dropout
# 测试的时候用model.eval(),不做BN和Dropout
# 训练函数
def fit(steps, model, loss_func, opt, train_dl, valid_dl):
    for step in range(steps):
        # 开启训练模式
        model.train()
        for xb, yb in train_dl:
            loss_batch(model, loss_func, xb, yb, opt)
        # 开启验证模式
        model.eval()
        with torch.no_grad():
            losses, nums = zip(
                *[loss_batch(model, loss_func, xb, yb) for xb, yb in valid_dl]
            )
        # 计算平均损失
        val_loss = np.sum(np.multiply(losses, nums)) / np.sum(nums)
        print('当前step:' + str(step), '验证集损失：' + str(val_loss))


# 获取模型和优化器
def get_model():
    model = MnistNn()
    # SGD优化器
    return model, optim.SGD(model.parameters(), lr=0.001)


# 进行一次迭代
def loss_batch(model, loss_func, xb, yb, opt=None):
    loss = loss_func(model(xb), yb)

    # 如果有优化器,就进行反向传播,做梯度下降优化
    if opt is not None:
        # 反向传播计算梯度
        loss.backward()
        # 优化器执行一次参数更新
        opt.step()
        # 将梯度累计值清零
        opt.zero_grad()

    return loss.item(), len(xb)


# 获取数据
def get_data(train_ds, valid_ds, bs):
    return (
        DataLoader(train_ds, batch_size=bs, shuffle=True),
        DataLoader(valid_ds, batch_size=bs * 2),
    )


train_dl, valid_dl = get_data(train_ds, valid_ds, bs)
model, opt = get_model()
fit(25, model, loss_func, opt, train_dl, valid_dl)

# 训练完模型,看下准确率
correct = 0
total = 0
# 一个batch一个batch的去看
for xb, yb in valid_dl:
    outputs = model(xb)
    # 最大的值,和对应索引,这里只要索引
    _, predicted = torch.max(outputs, 1)
    total += yb.size(0)
    correct += (predicted == yb).sum().item()
print('accuracy is: %d %%' % (100*correct/total))
