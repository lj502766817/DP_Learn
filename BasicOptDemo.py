"""
tensorflow的一些基础操作
"""
import tensorflow as tf
import numpy as np

print(tf.__version__)

x = [[2, 3]]
y = [[4], [5]]
# 矩阵乘法
z = tf.matmul(x, y)
print(z)
print('--------------')
# 构造一个Tensor
x = tf.constant([[1, 9], [3, 6]])
print(x)
print('--------------')
# 张量的加法,支持广播
x = tf.add(x, 1)
print(x)
print('--------------')
# 格式的转换
x = tf.cast(x, tf.float32)
print(x.numpy())
x1 = np.ones([2, 2])
x2 = tf.multiply(x1, 2)
print(x2)
print('--------------')
# 按第一个维度切分数据
input_data = np.arange(5)
dataset = tf.data.Dataset.from_tensor_slices(input_data)
for data in dataset:
    print(data)
print('--------------')
# 复制数据,使原始数据出现两次
dataset2 = dataset.repeat(2)
for data in dataset2:
    print(data)
print('--------------')
# 分组打包数据,4个一组
dataset_batch = dataset.repeat(2).batch(4)
for data in dataset_batch:
    print(data)
print('--------------')
