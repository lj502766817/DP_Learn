"""
基础的回归任务
"""
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
from sklearn import preprocessing

# 读取数据
features = pd.read_csv('./data/temps.csv')
print(features.head())
print('数据维度: ', features.shape)

# 为了画图,处理下时间数据
# 分别得到年，月，日
years = features['year']
months = features['month']
days = features['day']
# datetime格式
dates = [str(int(year)) + '-' + str(int(month)) + '-' + str(int(day)) for year, month, day in zip(years, months, days)]
dates = [datetime.datetime.strptime(date, '%Y-%m-%d') for date in dates]
# 画图看一下数据
# 指定默认风格
plt.style.use('fivethirtyeight')
# 设置布局
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2, figsize=(10, 10))
fig.autofmt_xdate(rotation=45)
# 标签值
ax1.plot(dates, features['actual'])
ax1.set_xlabel('')
ax1.set_ylabel('Temperature')
ax1.set_title('Max Temp')
# 昨天
ax2.plot(dates, features['temp_1'])
ax2.set_xlabel('')
ax2.set_ylabel('Temperature')
ax2.set_title('yesterday Max Temp')
# 前天
ax3.plot(dates, features['temp_2'])
ax3.set_xlabel('Date')
ax3.set_ylabel('Temperature')
ax3.set_title('the day before yesterday Max Temp')
# 朋友猜的
ax4.plot(dates, features['friend'])
ax4.set_xlabel('Date')
ax4.set_ylabel('Temperature')
ax4.set_title('Friend Estimate')
plt.tight_layout(pad=2)
plt.show()

# 独热编码,将week的数据变成one-hot的形式,不然网络做不了
features = pd.get_dummies(features)
print(features.head(5))
# 把特征和标签分开
# 标签
labels = np.array(features['actual'])
# 在特征中去掉标签
features = features.drop('actual', axis=1)
features = np.array(features)
print('最终的数据维度: ', features.shape)

# 用sklearn把数据做下预处理,后续可以用torchvision去做
input_features = preprocessing.StandardScaler().fit_transform(features)

# 开始构建网络模型,一个很基础很简单的例子
# 把数据转换成torch里的格式
x = torch.tensor(input_features, dtype=torch.float)
# 标签的shape和输入要保持一致
y = torch.tensor(labels, dtype=torch.float).view((-1, 1))

# 权重参数初始化
# 数据处理完后,一个样本是14个特征,第一层是就14*128
weights = torch.randn((14, 128), dtype=torch.float, requires_grad=True)
# 第一层偏置是128
biases = torch.randn(128, dtype=torch.float, requires_grad=True)
# 第二层得到结果就是128*1
weights2 = torch.randn((128, 1), dtype=torch.float, requires_grad=True)
# 结果只要一个偏置
biases2 = torch.randn(1, dtype=torch.float, requires_grad=True)
# 学习率
learning_rate = 0.001
# 损失值的统计
losses = []
# 迭代
for i in range(1000):
    # 计算隐层,y=xw+b
    hidden = x.mm(weights) + biases
    # 激活函数
    hidden = torch.relu(hidden)
    # 预测结果
    predictions = hidden.mm(weights2) + biases2
    # 通计算损失,均方差MSE
    loss = torch.mean((predictions - y) ** 2)
    losses.append(loss.data.numpy())

    # 打印损失值
    if i % 100 == 0:
        print('loss:', loss)
    # 返向传播计算
    loss.backward()

    # 更新参数,梯度下降方向更新
    weights.data.add_(- learning_rate * weights.grad.data)
    biases.data.add_(- learning_rate * biases.grad.data)
    weights2.data.add_(- learning_rate * weights2.grad.data)
    biases2.data.add_(- learning_rate * biases2.grad.data)

    # 在pytorch里梯度是默认累加的,所以每次迭代都要清空一下梯度
    weights.grad.data.zero_()
    biases.grad.data.zero_()
    weights2.grad.data.zero_()
    biases2.grad.data.zero_()

# 用一种更简单的方式构建网络
input_size = input_features.shape[1]
hidden_size = 128
output_size = 1
batch_size = 16
# 整体的网络结构和上面的一样,Sequential表示用一个顺序执行的结构
my_nn = torch.nn.Sequential(
    torch.nn.Linear(input_size, hidden_size),
    # 目前的样本量,激活函数换成Sigmoid也ok,不会梯度消失
    torch.nn.Sigmoid(),
    torch.nn.Linear(hidden_size, output_size),
)
# 损失函数还是用MSE
cost = torch.nn.MSELoss(reduction='mean')
# 通常情况下,无脑用Adam就完事了
optimizer = optim.Adam(my_nn.parameters(), lr=0.001)

# 训练网络
losses = []
# 做1000个epoch
for i in range(1000):
    batch_loss = []
    # 一个batch一个batch的取数据
    for start in range(0, len(input_features), batch_size):
        end = start + batch_size if start + batch_size < len(input_features) else len(input_features)
        xx = torch.tensor(input_features[start:end], dtype=torch.float, requires_grad=True)
        yy = torch.tensor(labels[start:end], dtype=torch.float, requires_grad=True).view((-1, 1))
        # 前向传播拿预测值
        prediction = my_nn(xx)
        # 计算损失
        loss = cost(prediction, yy)
        # 先清空下累计的梯度
        optimizer.zero_grad()
        # 反向传播计算梯度
        loss.backward(retain_graph=True)
        # 做参数优化
        optimizer.step()
        # 把损失存一下
        batch_loss.append(loss.data.numpy())

    # 打印损失
    if i % 100 == 0:
        # 当次的损失是每个样本的损失取平均
        losses.append(np.mean(batch_loss))
        print(i, np.mean(batch_loss))

# 做下预测
x = torch.tensor(input_features, dtype=torch.float)
predict = my_nn(x).data.numpy()

# 画个对比图看看
true_data = pd.DataFrame(data={'date': dates, 'actual': labels})
predictions_data = pd.DataFrame(data={'date': dates, 'prediction': predict.reshape(-1)})
# 真实值
plt.plot(true_data['date'], true_data['actual'], 'b-', label='actual')
# 预测值
plt.plot(predictions_data['date'], predictions_data['prediction'], 'ro', label='prediction')
plt.xticks(rotation='60')
plt.legend()
# 图名
plt.xlabel('Date')
plt.ylabel('Maximum Temperature (F)')
plt.title('Actual and Predicted Values')
# 看图...感觉过拟合了
plt.show()
