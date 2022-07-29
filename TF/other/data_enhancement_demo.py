"""
数据增强的一些操作
"""
import glob
import matplotlib.pyplot as plt
from PIL import Image
from tensorflow.keras.preprocessing import image


# 用来输出图像数据
def print_result(path):
    name_list = glob.glob(path)
    fig = plt.figure(figsize=(10, 14))
    for index in range(len(name_list)):
        img = Image.open(name_list[index])
        sub_img = fig.add_subplot(131 + index)
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
    # batch_size=1,那么每次迭代就会产生1个结果,也就是一个图片数据
    # 实际输出的是图片的像素值
    gen_data.next()
print_result(out_path + 'resize/*')

# 角度旋转
datagen = image.ImageDataGenerator(rotation_range=45)
gen_data = datagen.flow_from_directory(in_path, batch_size=1, shuffle=False, save_to_dir=out_path + 'rotation_range',
                                       save_prefix='gen', target_size=(224, 224))
for i in range(3):
    gen_data.next()
print_result(out_path + 'rotation_range/*')

# 平移变换
datagen = image.ImageDataGenerator(width_shift_range=0.3, height_shift_range=0.3)
gen_data = datagen.flow_from_directory(in_path, batch_size=1, shuffle=False, save_to_dir=out_path + 'shift',
                                       save_prefix='gen', target_size=(224, 224))
for i in range(3):
    gen_data.next()
print_result(out_path + 'shift/*')

# 放缩变换
datagen = image.ImageDataGenerator(zoom_range=0.5)
gen_data = datagen.flow_from_directory(in_path, batch_size=1, shuffle=False, save_to_dir=out_path + 'zoom',
                                       save_prefix='gen', target_size=(224, 224))
for i in range(3):
    gen_data.next()
print_result(out_path + 'zoom/*')

# 图像通道的平移,肉眼看貌似看不出来什么效果
datagen = image.ImageDataGenerator(channel_shift_range=15)
gen_data = datagen.flow_from_directory(in_path, batch_size=1, shuffle=False, save_to_dir=out_path + 'channel',
                                       save_prefix='gen', target_size=(224, 224))
for i in range(3):
    gen_data.next()
print_result(out_path + 'channel/*')

# 水平翻转
datagen = image.ImageDataGenerator(horizontal_flip=True)
gen_data = datagen.flow_from_directory(in_path, batch_size=1, shuffle=False, save_to_dir=out_path + 'horizontal',
                                       save_prefix='gen', target_size=(224, 224))
for i in range(3):
    gen_data.next()
print_result(out_path + 'horizontal/*')

# 像素值的归一化操作
datagen = image.ImageDataGenerator(rescale=1 / 255)
gen_data = datagen.flow_from_directory(in_path, batch_size=1, shuffle=False, save_to_dir=out_path + 'rescale',
                                       save_prefix='gen', target_size=(224, 224))
for i in range(3):
    gen_data.next()
print_result(out_path + 'rescale/*')

# 图像变换后,对空白的填充
# 填充方法
# 'constant': kkkkkkkk|abcd|kkkkkkkk (cval=k)
# 'nearest': aaaaaaaa|abcd|dddddddd
# 'reflect': abcddcba|abcd|dcbaabcd
# 'wrap': abcdabcd|abcd|abcdabcd
datagen = image.ImageDataGenerator(fill_mode='wrap', zoom_range=[4, 4])
gen_data = datagen.flow_from_directory(in_path, batch_size=1, shuffle=False, save_to_dir=out_path + 'fill_mode',
                                       save_prefix='gen', target_size=(224, 224))
for i in range(3):
    gen_data.next()
print_result(out_path + 'fill_mode/*')
