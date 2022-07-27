"""
猫狗识别demo
"""
import os
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# 数据文件夹
base_dir = './data/cats_and_dogs'
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')
# 训练集
train_cats_dir = os.path.join(train_dir, 'cats')
train_dogs_dir = os.path.join(train_dir, 'dogs')
# 验证集
validation_cats_dir = os.path.join(validation_dir, 'cats')
validation_dogs_dir = os.path.join(validation_dir, 'dogs')

# 构建网络
model = tf.keras.models.Sequential([
    # 第一层32个卷积核,卷积核大小是3*3*3,输入的数据是64*64*3
    # 其他参数先用默认值,默认步长是(1,1),那么第一层的特征图长宽就是(64-3)/1+1=62,结果就是(62,62,32)
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
    # 池化层取对半
    tf.keras.layers.MaxPooling2D(2, 2),

    # 因为之前池化层取了对半,丢了很多特征,所以这一层卷积核翻一倍做下弥补,
    # 这一层卷积核的大小是3*3*32,因为前一层获得了32个特征图
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),

    # 为全连接层准备,拉平数据
    tf.keras.layers.Flatten(),
    # FC,
    tf.keras.layers.Dense(512, activation='relu'),
    # 二分类可以直接用sigmoid就行
    tf.keras.layers.Dense(1, activation='sigmoid')
])
