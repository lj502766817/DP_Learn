"""
分类任务demo
"""
import pickle
import gzip
from pathlib import Path
from matplotlib import pyplot
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

# 读mnist数据,mnist数据集是28*28*1的灰度图,每个数据有784个特征,就是784个像素点
DATA_PATH = Path("data")
PATH = DATA_PATH / "mnist"
FILENAME = "mnist.pkl.gz"
with gzip.open((PATH / FILENAME).as_posix(), "rb") as f:
    ((x_train, y_train), (x_valid, y_valid), _) = pickle.load(f, encoding="latin-1")
print(x_train.shape)
# 弄一个数据图像看看
# pyplot.imshow(x_train[0].reshape((28, 28)), cmap="gray")
# pyplot.show()
# print(y_train[0])

# 分类任务的基本原理看这,https://github.com/lj502766817/ML_Algorithm/tree/main/LogisticRegression
# 大概说下就是样本的784个像素值,经过隐层的fc特征提取,然后得到10个分类分数,然后经过softmax函数得到各个类别的概率
model = tf.keras.Sequential()
model.add(layers.Dense(32, activation='relu'))
model.add(layers.Dense(32, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))
# 选择损失和评估函数要用合适的
model.compile(
    # adam优化算法,https://blog.csdn.net/dianyanxia/article/details/107862618
    optimizer=tf.keras.optimizers.Adam(0.001),
    # 交叉熵损失函数,注意有多种实现:https://www.tensorflow.org/api_docs/python/tf/keras/losses/SparseCategoricalCrossentropy
    loss=tf.losses.SparseCategoricalCrossentropy(),
    # 真实值与预测值匹配率函数:acc = np.dot(sample_weight, np.equal(y_true, np.argmax(y_pred, axis=1))
    metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]
)
# 训练
model.fit(x_train, y_train, epochs=5, batch_size=64, validation_data=(x_valid, y_valid))
