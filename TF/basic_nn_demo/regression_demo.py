"""
回归任务demo
"""
import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn import preprocessing
from tensorflow.keras import layers

# 先读取数据,看数据是什么样的
features = pd.read_csv("./data/temps.csv")
# print(features.head())

# 分析特征
# year,moth,day,week分别表示的具体的时间
# temp_2：前天的最高温度值
# temp_1：昨天的最高温度值
# average：在历史中，每年这一天的平均最高温度值
# actual：这就是我们的标签值了，当天的真实最高温度
# friend: 朋友瞎猜的值,没什么用,可以忽略

# 处理时间数据
# 得到年，月，日
years = features['year']
months = features['month']
days = features['day']
# 做成datetime格式
dates = [str(int(year)) + '-' + str(int(month)) + '-' + str(int(day)) for year, month, day in zip(years, months, days)]
dates = [datetime.datetime.strptime(date, '%Y-%m-%d') for date in dates]

# # 画图看看数据
# # 指定默认风格
# plt.style.use('fivethirtyeight')
# # 设置布局
# fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2, figsize=(10, 10))
# fig.autofmt_xdate(rotation=45)
# # 标签值
# ax1.plot(dates, features['actual'])
# ax1.set_xlabel('')
# ax1.set_ylabel('Temperature')
# ax1.set_title('Today Max Temp')
# # 昨天
# ax2.plot(dates, features['temp_1'])
# ax2.set_xlabel('')
# ax2.set_ylabel('Temperature')
# ax2.set_title('Yesterday Max Temp')
# # 前天
# ax3.plot(dates, features['temp_2'])
# ax3.set_xlabel('Date')
# ax3.set_ylabel('Temperature')
# ax3.set_title('The Day Before Yesterday Max Temp')
# # 朋友猜的
# ax4.plot(dates, features['friend'])
# ax4.set_xlabel('Date')
# ax4.set_ylabel('Temperature')
# ax4.set_title('Friend Estimate')
#
# plt.tight_layout(pad=2)
# plt.show()

# 将特征做一下one-hot热编码
features = pd.get_dummies(features)
# print(features.head(5))
# 取出标签值
labels = np.array(features['actual'])
# 在特征中去掉标签
features = features.drop('actual', axis=1)
# 转换格式
features = np.array(features)
# print(features.shape)

# 用sklearn做下数据的预处理,直接默认处理下
input_features = preprocessing.StandardScaler().fit_transform(features)
# print(input_features[0])


# 基于Keras构建网络模型 列一些常用的参数
# activation：激活函数的选择，一般常用relu
# kernel_initializer,bias_initializer：权重与偏置参数的初始化方法，有的时候初始化的好就很快能够很好的收敛,这个看脸
# kernel_regularizer，bias_regularizer：正则化惩罚，
# inputs：输入，可以自己指定，也可以让网络自动选
# units：神经元个数

# 按顺序构造网络的模型
model = tf.keras.Sequential()
# 这一层用16个神经元
model.add(layers.Dense(16))
model.add(layers.Dense(32))
model.add(layers.Dense(1))
# 指定优化器和损失函数,小批量梯度下降(直译是随机梯度下降)和均方误差
model.compile(optimizer=tf.keras.optimizers.SGD(0.001), loss='mean_squared_error')
# 进行训练
model.fit(input_features, labels, validation_split=0.25, epochs=10, batch_size=64)
# 收敛不好,看一下模型的概览
# print(model.summary())

# 换一下参数初始化的方法看看效果
print("\n change kernel_initializer \n")
model = tf.keras.Sequential()
model.add(layers.Dense(16, kernel_initializer='random_normal'))
model.add(layers.Dense(32, kernel_initializer='random_normal'))
model.add(layers.Dense(1, kernel_initializer='random_normal'))
model.compile(optimizer=tf.keras.optimizers.SGD(0.001), loss='mean_squared_error')
model.fit(input_features, labels, validation_split=0.25, epochs=100, batch_size=64)

# 在加入正则化惩罚看看,loss = l2 * reduce_sum(square(x))
print("\n add kernel_regularizer \n")
model = tf.keras.Sequential()
model.add(layers.Dense(16, kernel_initializer='random_normal', kernel_regularizer=tf.keras.regularizers.l2(0.03)))
model.add(layers.Dense(32, kernel_initializer='random_normal', kernel_regularizer=tf.keras.regularizers.l2(0.03)))
model.add(layers.Dense(1, kernel_initializer='random_normal', kernel_regularizer=tf.keras.regularizers.l2(0.03)))
model.compile(optimizer=tf.keras.optimizers.SGD(0.001), loss='mean_squared_error')
model.fit(input_features, labels, validation_split=0.25, epochs=100, batch_size=64)

# 用模型做预测
predict = model.predict(input_features)

# 画预测值和真实值的对比图
# 真实值的数据
true_data = pd.DataFrame(data={'date': dates, 'actual': labels})
# 预测值的数据,直接把预测值拉平
predictions_data = pd.DataFrame(data={'date': dates, 'prediction': predict.reshape(-1)})
# 对比
# 真实值
plt.plot(true_data['date'], true_data['actual'], 'b-', label='actual')
# 预测值
plt.plot(predictions_data['date'], predictions_data['prediction'], 'ro', label='prediction')
plt.xticks(rotation='60')
plt.legend()
# 一些标注
plt.xlabel('Date')
plt.ylabel('Maximum Temperature (F)')
plt.title('Actual and Predicted Values')
plt.show()
