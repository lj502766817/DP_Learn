"""
lstm的文本分类任务
nlp任务因为我们的输入是文字,但是神经网络的输入要是一个矩阵
那么首先要做的就是做分词,可以按词分或者按字分都行
然后将我们数据中的词按照一个语料表映射成一个个id,例如分字情况下:我喝咖啡就映射成了[2,9,4,8]
然后在做一个embedding(词嵌入),根据把id再映射成一个个向量,
比如一个字是300维的向量,那么最后我们一句话就变成了一个(4,300)的矩阵,就能放到神经网络里了
"""
import time
import torch
import numpy as np
from train_eval import train, init_network
from importlib import import_module
import argparse
from tensorboardX import SummaryWriter

# 启动参数 --model TextRNN
parser = argparse.ArgumentParser(description='Chinese Text Classification')
parser.add_argument('--model', type=str, required=True,
                    help='choose a model: TextCNN, TextRNN, FastText, TextRCNN, TextRNN_Att, DPCNN, Transformer')
parser.add_argument('--embedding', default='pre_trained', type=str, help='random or pre_trained')
parser.add_argument('--word', default=False, type=bool, help='True for word, False for char')
args = parser.parse_args()

if __name__ == '__main__':
    # 数据的位置
    dataset = 'THUCNews'

    # embedding表我们一般搞不出来,就直接拿大厂的开源的来做
    # 默认用搜狗新闻开源的
    embedding = 'embedding_SougouNews.npz'
    if args.embedding == 'random':
        embedding = 'random'
    # TextCNN, TextRNN 现在自己写的就这两个,用TextRNN
    model_name = args.model
    # 根据不同的模型去加载不同的类
    if model_name == 'FastText':
        from utils_fasttext import build_dataset, build_iterator, get_time_dif
    else:
        from utils import build_dataset, build_iterator, get_time_dif

    # import_module,动态导入需要的python程序文件的module
    x = import_module('models.' + model_name)
    # 初始化模型的配置
    config = x.Config(dataset, embedding)
    # 把一些随机值的种子设置成1,让每次随机初始值一样
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True  # 保证每次结果一样

    start_time = time.time()
    print("Loading data...")
    # 构建数据集
    vocab, train_data, dev_data, test_data = build_dataset(config, args.word)
    # 构建数据迭代器,dataloader
    train_iter = build_iterator(train_data, config)
    dev_iter = build_iterator(dev_data, config)
    test_iter = build_iterator(test_data, config)
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)

    # train
    config.n_vocab = len(vocab)
    # 构建模型并丢到设备里
    model = x.Model(config).to(config.device)
    # 一些日志
    writer = SummaryWriter(log_dir=config.log_path + '/' + time.strftime('%m-%d_%H.%M', time.localtime()))
    if model_name != 'Transformer':
        init_network(model)
    print(model.parameters)
    # 做训练
    train(config, model, train_iter, dev_iter, test_iter, writer)
