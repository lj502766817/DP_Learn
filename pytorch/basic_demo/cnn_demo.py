"""
一个简单的卷积神经网络例子
"""
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as torch_data_util
from torchvision import datasets, transforms

# 先把一些超参数定义一下
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")
# 图像的总尺寸28*28
input_size = 28
# 标签的种类数
num_classes = 10
# 训练的总循环周期
num_epochs = 3
# 一个批次的大小，64张图片
batch_size = 64

# 用内置的mnist数据集
# 读数据
# 训练集
train_dataset = datasets.MNIST(root='./data',
                               train=True,
                               transform=transforms.ToTensor(),
                               download=True)

# 测试集
test_dataset = datasets.MNIST(root='./data',
                              train=False,
                              transform=transforms.ToTensor())
# 构建batch数据,方便后面一个batch一个batch的去加载数据
train_loader = torch_data_util.DataLoader(dataset=train_dataset,
                                          batch_size=batch_size,
                                          shuffle=True)
test_loader = torch_data_util.DataLoader(dataset=test_dataset,
                                         batch_size=batch_size,
                                         shuffle=True)


# 构建一个卷积模型
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # 通常做卷积是一个卷积模块一个卷积模块的做
        # 输入大小 (1, 28, 28),pytorch是channel first的,即把通道数放前面
        # 这一个模块显示对输入做一次卷积,然后rule激活,然后池化压缩特征
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                # mnist是灰度图,就一个通道
                in_channels=1,
                # 卷积核的个数
                out_channels=16,
                # 卷积核大小
                kernel_size=5,
                # 步长
                stride=1,
                # 如果希望卷积后大小跟原来一样，需要设置padding=(kernel_size-1)/2 if stride=1
                padding=2,
            ),
            # 输出的特征图为 (16, 28, 28),输出特征图的大小是(h/w-kernel_size+2*padding)/stride+1
            # relu层
            nn.ReLU(),
            # 进行池化操作（2x2 区域）取对半, 输出结果为： (16, 14, 14)
            nn.MaxPool2d(kernel_size=2),
        )
        # 下一个模块的输入 (16, 14, 14),这一个模块做了两次卷积操作
        self.conv2 = nn.Sequential(
            # 卷积操作输出 (32, 14, 14)
            nn.Conv2d(16, 32, 5, 1, 2),
            nn.ReLU(),
            nn.Conv2d(32, 32, 5, 1, 2),
            nn.ReLU(),
            # 池化操作输出 (32, 7, 7)
            nn.MaxPool2d(2),
        )

        # 下一个模块的输入 (32, 7, 7)
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, 5, 1, 2),
            nn.ReLU(),
            # 单纯卷积和激活函数,输出 (32, 7, 7)
        )

        # 最后一个全连接得到10分类的结果
        self.out = nn.Linear(64 * 7 * 7, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        # flatten操作，把特征图拉平成一个特征向量,最后用来做分类或者回归等任务,结果为：(batch_size, 32 * 7 * 7)
        x = x.view(x.size(0), -1)
        output_val = self.out(x)
        return output_val


# 计算准确率的接口
def accuracy(predictions, labels):
    # 只需要序号就能表示预测值的结果了
    pred = torch.max(predictions.data, 1)[1]
    # 把labels的格式调整成pred一样,然后找匹配的
    rights = pred.eq(labels.data.view_as(pred)).sum()
    # 返回正确的个数和总样本数
    return rights, len(labels)


# 做模型的训练
# 实例化
net = CNN().to(device)
# 损失函数,分类任务用交叉熵
criterion = nn.CrossEntropyLoss()
# 优化器
optimizer = optim.Adam(net.parameters(), lr=0.001)

# 开始训练循环
for epoch in range(num_epochs):
    # 当前epoch的结果保存下来
    train_rights = []

    # 针对容器中的每一个批进行循环
    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.to(device)
        target = target.to(device)
        # 开启训练模式
        net.train()
        # 前向传播计算结果
        output = net(data)
        # 计算损失
        loss = criterion(output, target)
        # 清空累计梯度
        optimizer.zero_grad()
        # 反向传播计算梯度
        loss.backward()
        # 优化参数
        optimizer.step()
        # 计算正确率
        right = accuracy(output, target)
        # 保存记录
        train_rights.append(right)

        # 每100个批次看一下验证集的正确率
        if batch_idx % 100 == 0:
            # 开启验证模式
            net.eval()
            val_rights = []

            for (test_data, test_target) in test_loader:
                test_data = test_data.to(device)
                test_target = test_target.to(device)
                output = net(test_data)
                right = accuracy(output, test_target)
                val_rights.append(right)

            # 准确率计算
            train_r = (sum([tup[0] for tup in train_rights]), sum([tup[1] for tup in train_rights]))
            val_r = (sum([tup[0] for tup in val_rights]), sum([tup[1] for tup in val_rights]))

            print('当前epoch: {} [{}/{} ({:.0f}%)]\t当前损失: {:.6f}\t训练集准确率: {:.2f}%\t测试集正确率: {:.2f}%'
                  .format(epoch, batch_idx * batch_size, len(train_dataset)
                          , 100. * batch_idx / len(train_loader)
                          , loss.data, 100. * train_r[0] / train_r[1], 100. * val_r[0] / val_r[1]))
