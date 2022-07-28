"""
数据增强的一些操作
"""
import glob
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from tensorflow.keras.preprocessing import image


# 用来输出图像数据
def print_result(path):
    name_list = glob.glob(path)
    fig = plt.figure(figsize=(12, 16))
    for i in range(len(name_list)):
        img = Image.open(name_list[i])
        sub_img = fig.add_subplot(131 + i)
        sub_img.imshow(img)
    plt.show()


out_path = './out/'
in_path = './data/img/'
img_path = './data/img/superman/*'
print_result(img_path)

# 尺寸变换
datagen = image.ImageDataGenerator()
gen_data = datagen.flow_from_directory(in_path, batch_size=1, shuffle=False,
                                       save_to_dir=out_path + 'resize',
                                       save_prefix='gen', target_size=(224, 224))
for i in range(3):
    gen_data.next()
print_result(out_path + 'resize/*')

# 角度旋转
datagen = image.ImageDataGenerator(rotation_range=45)
gen = image.ImageDataGenerator()
data = gen.flow_from_directory(in_path, batch_size=1, class_mode=None, shuffle=True, target_size=(224, 224))
np_data = np.concatenate([data.next() for i in range(data.n)])
datagen.fit(np_data)
gen_data = datagen.flow_from_directory(in_path, batch_size=1, shuffle=False, save_to_dir=out_path + 'rotation_range',
                                       save_prefix='gen', target_size=(224, 224))
for i in range(3):
    gen_data.next()
print_result(out_path + 'rotation_range/*')
