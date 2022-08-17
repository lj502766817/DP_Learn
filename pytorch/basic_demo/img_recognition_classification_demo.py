"""
图像识别分类的简单例子
"""
import os
import time
import copy

import torch
import json
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
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
            # 把图片大小压缩到l*w
            transforms.Resize([96, 96]),
            # 随机旋转，-45到45度之间随机选
            transforms.RandomRotation(45),
            # 在中心位置裁剪x*x的大小
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
            # (x-μ)/σ,减均值除标准差
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
# print("image_datasets:")
# print(image_datasets)
# print()
# print("class_names:")
# print(class_names)
# print("size:", dataset_sizes)

# 读取标签实际对应的名字
with open('./data/flower_data/cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)
# print(cat_to_name)

# 进行迁移学习,这里用resnet来做
# 可选的比较多 ['resnet', 'alexnet', 'vgg', 'squeezenet', 'densenet', 'inception']
model_name = 'resnet'
# 是否用人家训练好的特征来做,都用人家特征，先不更新
feature_extract = True

# 是用GPU还是用CPU来训练,比较通用的代码
train_on_gpu = torch.cuda.is_available()
if not train_on_gpu:
    print('CUDA is not available.  Training on CPU ...')
else:
    print('CUDA is available!  Training on GPU ...')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# # pytorch内置的比较经典的有18,34,50,101,152
# model_ft = models.resnet50()
# # 可以打印下看看网络结构
# print(model_ft)


# 设置模型参数是否需要更新的方法
def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


# 初始化模型,因为是做迁移学习,并且resnet原来是一个1000分类的任务
# 我们可以先将resnet前面的结构冻住,不更新参数,只在最后一层的fc去训练我们自己的参数,进行一个102分类的任务
def initialize_model(model_name, num_classes, feature_extract, use_pretrained=True):
    # 先获取一个resnet模型
    model_ft = models.resnet18(pretrained=use_pretrained)
    # 然后设置模型的参数是否要更新
    set_parameter_requires_grad(model_ft, feature_extract)
    # 这里获得了模型最后一层fc层的入参维度
    num_ftrs = model_ft.fc.in_features
    # 类别数自己根据自己任务来,这里我们是102分类,那么自己的全连接层的输出就是102,用自己的fc层去替代resnet50原来的fc层
    # 并且这个fc层是自己定义的,所以权重参数没有被冻住,是可以更新的
    model_ft.fc = nn.Linear(num_ftrs, num_classes)
    # 输入大小根据自己配置来
    input_size = 64

    return model_ft, input_size


model_ft, input_size = initialize_model(model_name, 102, feature_extract, use_pretrained=True)

# GPU还是CPU计算
model_ft = model_ft.to(device)
# 模型保存，名字随便
filename = './out/best.pt'
# 获取需要更新的权重参数
params_to_update = []
print("Params to learn:")
for name, param in model_ft.named_parameters():
    if param.requires_grad:
        params_to_update.append(param)
        print("\t", name)
# 可以打印看看加入了自己的输出层之后的网络结构
print(model_ft)

# 优化器设置
# 传入需要训练的参数和学习率
optimizer_ft = optim.Adam(params_to_update, lr=1e-2)
# 学习率每10个epoch衰减成原来的1/10
scheduler = optim.lr_scheduler.StepLR(optimizer_ft, step_size=10, gamma=0.1)
# 交叉熵损失函数
criterion = nn.CrossEntropyLoss()


def train_model(model, dataloaders, criterion, optimizer, num_epochs=25, filename='best.pt'):
    # 做下时间记录
    since = time.time()
    # 学的最好的那次的准确率
    best_acc = 0
    # 把模型放到GPU或者CPU里去
    model.to(device)
    # 训练过程中打印各种损失和指标
    val_acc_history = []
    train_acc_history = []
    train_losses = []
    valid_losses = []
    # 拿到当前优化器的学习率
    LRs = [optimizer.param_groups[0]['lr']]
    # 初始化最好的那次模型
    best_model_wts = copy.deepcopy(model.state_dict())
    # 一个个epoch来做
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # 训练和验证一个个的来做
        for phase in ['train', 'valid']:
            if phase == 'train':
                # 切训练模式
                model.train()
            else:
                # 切验证
                model.eval()
            # 跑的过程中的损失和正确率
            running_loss = 0.0
            running_corrects = 0

            # 一个batch一个batch的去取数据,dataloaders[phase]得到的对应训练集或者验证集的dataloader
            for inputs, labels in dataloaders[phase]:
                # 要把数据放到CPU或GPU
                inputs = inputs.to(device)
                labels = labels.to(device)

                # 每次都要清零累计的梯度
                optimizer.zero_grad()
                # 前向传播计算结果
                outputs = model(inputs)
                # 计算损失
                loss = criterion(outputs, labels)
                # 因为是102分类,这里的output的结果是102分类对应的概率值,通过max得到概率最大的那个类别
                _, preds = torch.max(outputs, 1)
                # 训练阶段更新权重
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                # 计算损失
                # 总体的损失值
                # 0表示batch那个维度
                running_loss += loss.item() * inputs.size(0)
                # 总体的正确个数
                # 预测结果最大的和真实值是否一致
                running_corrects += torch.sum(preds == labels.data)

            # 算平均
            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            # 看看一个epoch花了多久
            time_elapsed = time.time() - since
            print('Time elapsed {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # 得到最好那次的模型
            if phase == 'valid' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                # 字典里key就是各层的名字，值就是训练好的权重
                state = {
                    'state_dict': model.state_dict(),
                    'best_acc': best_acc,
                    'optimizer': optimizer.state_dict(),
                }
                torch.save(state, filename)
            # 存下记录
            if phase == 'valid':
                val_acc_history.append(epoch_acc)
                valid_losses.append(epoch_loss)
            if phase == 'train':
                train_acc_history.append(epoch_acc)
                train_losses.append(epoch_loss)
                # 训练的时候才做学习率衰减
                scheduler.step()

        print('Optimizer learning rate : {:.7f}'.format(optimizer.param_groups[0]['lr']))
        LRs.append(optimizer.param_groups[0]['lr'])
        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # 训练完后用最好的一次当做模型最终的结果,最后做测试用
    model.load_state_dict(best_model_wts)
    return model, val_acc_history, train_acc_history, valid_losses, train_losses, LRs


# 训练
model_ft, val_acc_history, train_acc_history, valid_losses, train_losses, LRs = train_model(model_ft, dataloaders,
                                                                                            criterion, optimizer_ft,
                                                                                            num_epochs=20,
                                                                                            filename=filename)

# 重新整体训练,因为前面的训练是只训练我们自定义的fc层,当fc层训练有一定结果后,我们可以将整个模型整体训练一下
# 就类似前面我们是借助resnet的先验去提取特征训练自定义的fc层,然后在fc层训练的差不多的情况下,重新训练整体,不需要resnet的先验了,重新用我们的数据去训练了
for param in model_ft.parameters():
    param.requires_grad = True

# 再继续训练所有的参数，学习率调小一点
optimizer = optim.Adam(model_ft.parameters(), lr=1e-3)
scheduler = optim.lr_scheduler.StepLR(optimizer_ft, step_size=10, gamma=0.1)

# 损失函数
criterion = nn.CrossEntropyLoss()

# 加载之前训练好的权重参数
# 加载的参数必须和模型,以及模型的输入参数格式对应起来
checkpoint = torch.load(filename)
best_acc = checkpoint['best_acc']
model_ft.load_state_dict(checkpoint['state_dict'])
# 重新整体训练
model_ft, val_acc_history, train_acc_history, valid_losses, train_losses, LRs = train_model(model_ft, dataloaders,
                                                                                            criterion, optimizer,
                                                                                            num_epochs=10,
                                                                                            filename=filename)

# 做测试
# 得到一个batch的测试数据,这里就用验证集去做做测试集了
dataiter = iter(dataloaders['valid'])
images, labels = dataiter.next()

model_ft.eval()

if train_on_gpu:
    output = model_ft(images.cuda())
else:
    output = model_ft(images)
_, preds_tensor = torch.max(output, 1)
# 在GPU里的数据要导到CPU才能转成ndarray的结构
preds = np.squeeze(preds_tensor.numpy()) if not train_on_gpu else np.squeeze(preds_tensor.cpu().numpy())


# 把tensor重新转成图像
def im_convert(tensor):
    image = tensor.to("cpu").clone().detach()
    image = image.numpy().squeeze()
    # pytorch里图像的格式是(c,l,w),所以这里转换的时候要把通道的维度放后面去就是(1,2,0)了
    image = image.transpose(1, 2, 0)
    # 数据预处理的后做归一化了,这里要还原回来
    image = image * np.array((0.229, 0.224, 0.225)) + np.array((0.485, 0.456, 0.406))
    image = image.clip(0, 1)

    return image


# 画图展示下
fig = plt.figure(figsize=(20, 20))
columns = 4
rows = 2
for idx in range(columns * rows):
    ax = fig.add_subplot(rows, columns, idx + 1, xticks=[], yticks=[])
    plt.imshow(im_convert(images[idx]))
    ax.set_title("{} ({})".format(cat_to_name[str(preds[idx])], cat_to_name[str(labels[idx].item())]),
                 color=("green" if cat_to_name[str(preds[idx])] == cat_to_name[str(labels[idx].item())] else "red"))
plt.show()
