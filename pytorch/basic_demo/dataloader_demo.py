"""
实际场景中使用dataloader的例子
"""
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image


# 先拆解制作一个dataloader需要哪些基本的步骤

# 加载标注文件
def load_annotations(ann_file):
    data_infos = {}
    with open(ann_file) as f:
        samples = [x.strip().split(' ') for x in f.readlines()]
        for filename, gt_label in samples:
            data_infos[filename] = np.array(gt_label, dtype=np.int64)
    return data_infos


# print(load_annotations('./data/flower_data/train.txt'))

# 然后把样本和标签分别放到两个集合里,pytorch的dataloader是建议做成list的,非要做成其他格式其实也行
img_label = load_annotations('./data/flower_data/train.txt')
image_name = list(img_label.keys())
label = list(img_label.values())

# print(image_name[:5])
# print(label[:5])

# 因为标注文件的样本就只是个图片文件的名字,要用的时候是需要转换成文件的路径的
data_dir = './data/flower_data/'
train_dir = data_dir + '/train_filelist'
valid_dir = data_dir + '/val_filelist'
image_path = [os.path.join(train_dir, img) for img in image_name]


# 实际去创建一个dataloader
# 先建一个我们自己任务的dataset
# __init__函数和__getitem__需要根据自己的任务重写
class FlowerDataset(Dataset):
    def __init__(self, root_dir, ann_file, transform=None):
        self.ann_file = ann_file
        self.root_dir = root_dir
        self.img_label = self.load_annotations()
        self.img = [os.path.join(self.root_dir, img) for img in list(self.img_label.keys())]
        self.label = [label for label in list(self.img_label.values())]
        # 参数预处理
        self.transform = transform

    def __len__(self):
        return len(self.img)

    # __getitem__函数是实际上dataloader从dataset里拿数据调用的函数
    # 这个idx就是dataloader来拿数据的时候传过来的index,因为dataloader可能做洗牌的操作
    def __getitem__(self, idx):
        image = Image.open(self.img[idx])
        label = self.label[idx]
        # 如果有预处理参数,就对样本做预处理
        if self.transform:
            image = self.transform(image)
        # torch里用的话要把其他格式的数据转成tensor
        label = torch.from_numpy(np.array(label))
        return image, label

    def load_annotations(self):
        data_infos = {}
        with open(self.ann_file) as f:
            samples = [x.strip().split(' ') for x in f.readlines()]
            for filename, gt_label in samples:
                data_infos[filename] = np.array(gt_label, dtype=np.int64)
        return data_infos


# 这个数据的预处理实际上是在__getitem__这里做的
data_transforms = {
    'train':
        transforms.Compose([
            transforms.Resize(64),
            # 随机旋转，-45到45度之间随机选
            transforms.RandomRotation(45),
            # 从中心开始裁剪
            transforms.CenterCrop(64),
            # 随机水平翻转 选择一个概率概率
            transforms.RandomHorizontalFlip(p=0.5),
            # 随机垂直翻转
            transforms.RandomVerticalFlip(p=0.5),
            # 参数1为亮度，参数2为对比度，参数3为饱和度，参数4为色相
            transforms.ColorJitter(brightness=0.2, contrast=0.1, saturation=0.1, hue=0.1),
            # 概率转换成灰度率，3通道就是R=G=B
            transforms.RandomGrayscale(p=0.025),
            # 把数据转换成tensor
            transforms.ToTensor(),
            # 均值，标准差
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    'valid':
        transforms.Compose([
            transforms.Resize(64),
            transforms.CenterCrop(64),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
}

# 实际去试一下
train_dataset = FlowerDataset(root_dir=train_dir, ann_file='./data/flower_data/train.txt',
                              transform=data_transforms['train'])
val_dataset = FlowerDataset(root_dir=valid_dir, ann_file='./data/flower_data/val.txt',
                            transform=data_transforms['valid'])
# dataloader里的数据是都已经处理好了的,是直接往CPU或者GPU里发的
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=True)

# 拿个loader里的数据看看
image, label = iter(train_loader).next()
# 这个是把多余的维度去掉,因为从loader里取的是batch数据,维度是(1,3,64,64)这样的,要展示图片就要把前面这个1去掉
sample = image[0].squeeze()
# 把channel挪到后面去
sample = sample.permute((1, 2, 0)).numpy()
# 这里把归一化的数据再还原回去
sample *= [0.229, 0.224, 0.225]
sample += [0.485, 0.456, 0.406]
plt.imshow(sample)
plt.show()
print('Label is: {}'.format(label[0].numpy()))
