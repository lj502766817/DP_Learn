"""
图像识别分类的简单例子
"""
import os
import torch
import torch.utils.data as torch_data_util
from torchvision import transforms, models, datasets

# 数据路径地址
data_dir = './data/flower_data/'
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'

# 数据的预处理和增强
data_transforms = {
    'train':
        transforms.Compose([
            # 把图片大小压缩到96*96
            transforms.Resize([96, 96]),
            # 随机旋转，-45到45度之间随机选
            transforms.RandomRotation(45),
            # 在中心位置裁剪64*64的大小
            transforms.CenterCrop(64),
            # 随机水平翻转 选择一个概率概率
            transforms.RandomHorizontalFlip(p=0.5),
            # 随机垂直翻转
            transforms.RandomVerticalFlip(p=0.5),
            # 参数1为亮度，参数2为对比度，参数3为饱和度，参数4为色相
            transforms.ColorJitter(brightness=0.2, contrast=0.1, saturation=0.1, hue=0.1),
            # 概率转换成灰度率，3通道就是R=G=B
            transforms.RandomGrayscale(p=0.025),
            # 转换成tensor格式
            transforms.ToTensor(),
            # (x-μ)/σ^2,减均值除标准差
            # 这里用的先验数据
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    'valid':
        transforms.Compose([
            transforms.Resize([64, 64]),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
}

batch_size = 128
# 加载数据,因为数据都按文件夹分类好了,这里就简单处理用ImageFolder去加载数据,分别按train和valid文件夹去加载,做成一个字典的格式
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'valid']}
# 构建loader,也做成字典形式
dataloaders = {x: torch_data_util.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True) for x in
               ['train', 'valid']}
# 看一下训练集和验证集的数据量
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'valid']}
# 样本的labels,这里的排序方式并不是自然排序
class_names = image_datasets['train'].classes

# 打印看看
print("image_datasets:")
print(image_datasets)
print()
print("class_names:")
print(class_names)
print("size:", dataset_sizes)
