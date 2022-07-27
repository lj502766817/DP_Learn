"""
猫狗识别demo
"""
import os
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

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
    # 其他的什么padding,bias,正则化惩罚的先不管
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
    # FC
    tf.keras.layers.Dense(512, activation='relu'),
    # 二分类可以直接用sigmoid就行
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# 看下网络的整个结构
print(model.summary())

# 配置响应的训练参数
model.compile(
    # 二分类的交叉熵损失函数
    loss='binary_crossentropy',
    optimizer=Adam(lr=1e-4),
    # When you pass the strings 'accuracy' or 'acc',
    # we convert this to one of `tf.keras.metrics.BinaryAccuracy`,
    # `tf.keras.metrics.CategoricalAccuracy`,`tf.keras.metrics.SparseCategoricalAccuracy`
    # based on the loss function used and the model output shape.
    # We do a similar conversion for the strings 'crossentropy' and 'ce' as well.
    metrics=['acc']
)

# 对数据做预处理,图像被被读取成tensor(float32)格式,然后被归一化到(0,1)里面
# 新的官网文档有更推荐的方式,用tf.keras.utils.image_dataset_from_directory去读数据,然后做预处理
train_datagen = ImageDataGenerator(rescale=1. / 255)
test_datagen = ImageDataGenerator(rescale=1. / 255)
# 预处理后的训练集
train_generator = train_datagen.flow_from_directory(
    # 文件夹路径
    train_dir,
    # 指定resize成的大小
    target_size=(64, 64),
    batch_size=20,
    # 如果one-hot就是categorical，二分类用binary就可以
    class_mode='binary')
# 预处理后的验证集
validation_generator = test_datagen.flow_from_directory(
    validation_dir,
    target_size=(64, 64),
    batch_size=20,
    class_mode='binary')

# 进行训练
history = model.fit_generator(
    train_generator,
    steps_per_epoch=100,  # 2000 images = batch_size * steps
    epochs=20,
    validation_data=validation_generator,
    validation_steps=50,  # 1000 images = batch_size * steps
    verbose=2)

# 画图看看模型的效果
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'bo', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training Loss')
plt.plot(epochs, val_loss, 'b', label='Validation Loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()
# 完全过拟合了呢~
