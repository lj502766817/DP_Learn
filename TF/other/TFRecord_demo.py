"""
TFRecord数据集制作
TFRecord是TensorFlow官方推荐的一种{key,value}二进制序列化的数据集格式
制作好后就是一个二进制的文件,使用起来更加方便
"""
import tensorflow as tf
import numpy as np
import glob
import os


# tf.Example中可以使用以下几种格式：
# tf.train.BytesList: 可以使用的类型包括 string和byte
# tf.train.FloatList: 可以使用的类型包括 float和double
# tf.train.Int64List: 可以使用的类型包括 enum,bool, int32, uint32, int64

# 字节格式
def _bytes_feature(value):
    """Returns a bytes_list from a string/byte."""
    if isinstance(value, type(tf.constant(0))):
        # BytesList won't unpack a string from an EagerTensor.
        value = value.numpy()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


# 浮点数
def _float_feature(value):
    """Return a float_list form a float/double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


# 整数
def _int64_feature(value):
    """Return a int64_list from a bool/enum/int/uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


# 创建一个TFRecord数据集样本
def serialize_example(feature0, feature1, feature2, feature3):
    """
    创建tf.Example
    """

    # 转换成相应类型
    feature = {
        # 官方例子上好像有坑,这里不接受numpy.bool_的格式,要强制转成bool下
        'feature0': _int64_feature(bool(feature0)),
        'feature1': _int64_feature(feature1),
        'feature2': _bytes_feature(feature2),
        'feature3': _float_feature(feature3),
    }
    # 使用tf.train.Example来创建
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    # SerializeToString方法转换为二进制字符串
    return example_proto.SerializeToString()


# 数据量
n_observations = int(1e4)
# bool型特征
feature0 = np.random.choice([False, True], n_observations)
# 整数特征
feature1 = np.random.randint(0, 5, n_observations)
# 字符串特征,字符串用字节的形式表示
strings = np.array([b'cat', b'dog', b'chicken', b'horse', b'goat'])
feature2 = strings[feature1]
# 浮点数特征
feature3 = np.random.randn(n_observations)

# TFRecord数据集名称
filename = 'tfrecord_test'
# 生成一个TFRecord数据集
with tf.io.TFRecordWriter(filename) as writer:
    for i in range(n_observations):
        example = serialize_example(feature0[i], feature1[i], feature2[i], feature3[i])
        writer.write(example)

# 读取TFRecord数据集
filenames = [filename]
raw_dataset = tf.data.TFRecordDataset(filenames)
print(raw_dataset)

# 实际去制作一个图像的TFRecord数据集
# 标签
image_labels = {
    'dog': 0,
    'kangaroo': 1,
}
# 读数据，binary格式
image_string = open('./data/tfrecord_source/dog.jpg', 'rb').read()
label = image_labels['dog']


# 创建图像数据的Example
def image_example(image_string, label):
    image_shape = tf.image.decode_jpeg(image_string).shape

    feature = {
        'height': _int64_feature(image_shape[0]),
        'width': _int64_feature(image_shape[1]),
        'depth': _int64_feature(image_shape[2]),
        'label': _int64_feature(label),
        'image_raw': _bytes_feature(image_string),
    }

    return tf.train.Example(features=tf.train.Features(feature=feature))


# 打印部分信息出来看看Example里是些啥
image_example_proto = image_example(image_string, label)
for line in str(image_example_proto).split('\n')[:15]:
    print(line)

# 制作 `images.tfrecords`数据集
image_path = './data/tfrecord_source/'
images = glob.glob(image_path + '*.jpg')
record_file = 'images.tfrecord'
counter = 0

with tf.io.TFRecordWriter(record_file) as writer:
    for fname in images:
        with open(fname, 'rb') as f:
            image_string = f.read()
            label = image_labels[os.path.basename(fname).replace('.jpg', '')]

            # 生成一个`tf.Example`
            tf_example = image_example(image_string, label)

            # 将`tf.example` 写入 TFRecord
            writer.write(tf_example.SerializeToString())

            counter += 1
            print('Processed {:d} of {:d} images.'.format(
                counter, len(images)))

print(' Wrote {} images to {}'.format(counter, record_file))

# 加载一个制作好的TFRecord
raw_train_dataset = tf.data.TFRecordDataset('images.tfrecord')

# 因为之前制作的时候Example都是进过序列化了的,那么使用之前还要解析一下
# tf.io.parse_single_example(example_proto, feature_description)函数可以解析单条example

# 解析的格式需要跟之前创建example时一致
image_feature_description = {
    'height': tf.io.FixedLenFeature([], tf.int64),
    'width': tf.io.FixedLenFeature([], tf.int64),
    'depth': tf.io.FixedLenFeature([], tf.int64),
    'label': tf.io.FixedLenFeature([], tf.int64),
    'image_raw': tf.io.FixedLenFeature([], tf.string),
}


# 解析函数
def parse_tf_example(example_proto):
    # 解析Example
    parsed_example = tf.io.parse_single_example(example_proto, image_feature_description)

    # 预处理
    x_train = tf.image.decode_jpeg(parsed_example['image_raw'], channels=3)
    x_train = tf.image.resize(x_train, (416, 416))
    x_train /= 255.

    label_y = parsed_example['label']
    y_train = label_y

    return x_train, y_train


# 解析后的数据集
train_dataset = raw_train_dataset.map(parse_tf_example)

# 制作成训练集
train_ds = train_dataset.shuffle(buffer_size=10000).batch(2).repeat(10)
# 打印出来看看
for batch, (x, y) in enumerate(train_ds):
    print(batch, x.shape, y)
