"""
分类任务demo
"""
import pickle
import gzip
from pathlib import Path
from matplotlib import pyplot
import numpy as np
import tensorflow as tf
from tensorflow.python.keras import layers
from tensorflow import keras

# 解决CUDA_ERROR_OUT_OF_MEMORY: out of memory错误
physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

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
    optimizer=tf.keras.optimizers.Adam(0.005),
    # 交叉熵损失函数,注意有多种实现:https://www.tensorflow.org/api_docs/python/tf/keras/losses/SparseCategoricalCrossentropy
    loss=tf.losses.SparseCategoricalCrossentropy(),
    # 真实值与预测值匹配率函数:acc = np.dot(sample_weight, np.equal(y_true, np.argmax(y_pred, axis=1))
    metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]
)
# 训练
model.fit(x_train, y_train, epochs=5, batch_size=64, validation_data=(x_valid, y_valid))

# 重新整理数据,根据steps_per_epoch进行训练
train = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train = train.batch(32)
train = train.repeat()

valid = tf.data.Dataset.from_tensor_slices((x_valid, y_valid))
valid = valid.batch(32)
valid = valid.repeat()
print("user params steps_per_epoch")
model.fit(train, epochs=5, steps_per_epoch=100, validation_data=valid, validation_steps=100)

# 换个数据集练手
fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
# 数据集里有哪些类别
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
# 看一下数据
print(train_images.shape)
print(len(train_labels))
print(test_images.shape)
# 展示一下数据
pyplot.figure()
pyplot.imshow(train_images[0])
pyplot.colorbar()
pyplot.grid(False)
pyplot.show()
# 相当于做了个数据的预处理
train_images = train_images / 255.0
test_images = test_images / 255.0
pyplot.figure(figsize=(10, 10))
for i in range(25):
    pyplot.subplot(5, 5, i + 1)
    pyplot.xticks([])
    pyplot.yticks([])
    pyplot.grid(False)
    pyplot.imshow(train_images[i], cmap=pyplot.cm.binary)
    pyplot.xlabel(class_names[train_labels[i]])
pyplot.show()
# 做分类训练
model = keras.Sequential([
    # 这一步是把28*28像素的图片拉平
    keras.layers.Flatten(input_shape=(28, 28)),
    # 剩下的就正常做全连接,然后softmax做分类
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(train_images, train_labels, epochs=10)

# 模型训练好后,用测试集做下评估
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print('\nTest accuracy:', test_acc)
# 做下预测,看看预测结果
predictions = model.predict(test_images)
print(predictions[0])
print(np.argmax(predictions[0]))


# 做下结果的可视化展示
def plot_image(i, predictions_array, true_label, img):
    predictions_array, true_label, img = predictions_array, true_label[i], img[i]
    pyplot.grid(False)
    pyplot.xticks([])
    pyplot.yticks([])

    pyplot.imshow(img, cmap=pyplot.cm.binary)

    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'

    pyplot.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                            100 * np.max(predictions_array),
                                            class_names[true_label]),
                  color=color)


def plot_value_array(i, predictions_array, true_label):
    predictions_array, true_label = predictions_array, true_label[i]
    pyplot.grid(False)
    pyplot.xticks(range(10))
    pyplot.yticks([])
    thisplot = pyplot.bar(range(10), predictions_array, color="#777777")
    pyplot.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)

    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')


i = 0
pyplot.figure(figsize=(6, 3))
pyplot.subplot(1, 2, 1)
plot_image(i, predictions[i], test_labels, test_images)
pyplot.subplot(1, 2, 2)
plot_value_array(i, predictions[i], test_labels)
pyplot.show()

i = 12
pyplot.figure(figsize=(6, 3))
pyplot.subplot(1, 2, 1)
plot_image(i, predictions[i], test_labels, test_images)
pyplot.subplot(1, 2, 2)
plot_value_array(i, predictions[i], test_labels)
pyplot.show()

# 最后,训练好的网络需要保存下来
# 保存整个模型
model.save('fashion_model.h5')
# 把网络的架构导成json格式
config = model.to_json()
with open('config.json', 'w') as json:
    json.write(config)
# 从json里读取网络架构
model = keras.models.model_from_json(config)
print(model.summary())
# 查看下权重参数
weights = model.get_weights()
print(weights)
# 保存和加载权重参数
model.save_weights('weights.h5')
model.load_weights('weights.h5')
