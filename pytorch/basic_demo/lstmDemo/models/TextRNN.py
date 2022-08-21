# coding: UTF-8
import torch
import torch.nn as nn
import numpy as np


class Config(object):
    """配置参数"""

    def __init__(self, dataset, embedding):
        self.model_name = 'TextRNN'
        # 训练集位置
        self.train_path = dataset + '/data/train.txt'
        # 验证集位置
        self.dev_path = dataset + '/data/dev.txt'
        # 测试集位置
        self.test_path = dataset + '/data/test.txt'
        # 加载类别文件,这里就是个10分类
        self.class_list = [x.strip() for x in open(
            dataset + '/data/class.txt').readlines()]
        # 语料库位置
        self.vocab_path = dataset + '/data/vocab.pkl'
        # 保存模型的训练结果位置
        self.save_path = dataset + '/saved_dict/' + self.model_name + '.ckpt'
        # 一些日志的位置
        self.log_path = dataset + '/log/' + self.model_name
        # 如果指定了预训练词向量文件的话,就加载词向量
        self.embedding_pretrained = torch.tensor(
            np.load(dataset + '/data/' + embedding)["embeddings"].astype('float32')) \
            if embedding != 'random' else None
        # 设备,CPU或者GPU
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.dropout = 0.5
        # 若超过1000batch效果还没提升，则提前结束训练
        self.require_improvement = 1000
        # 类别数
        self.num_classes = len(self.class_list)
        # 词表大小，在运行时赋值
        self.n_vocab = 0
        self.num_epochs = 32
        self.batch_size = 128
        # 每句话处理成的长度,多退少补
        self.pad_size = 32
        self.learning_rate = 1e-3
        # 字向量维度, 若使用了预训练词向量，则维度统一,self.embedding_pretrained.size(1)就是取第二个维度的大小
        self.embed = self.embedding_pretrained.size(1) \
            if self.embedding_pretrained is not None else 300
        # lstm隐藏层数量,就是普通情况下,一个lstm层输出的向量维度
        self.hidden_size = 128
        # lstm层数,lstm可以层层累上去,第一层各个词的输出,可以作为第二层的输入
        self.num_layers = 3


'''Recurrent Neural Network for Text Classification with Multi-Task Learning'''


class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        # embedding层做词向量转换
        # 如果本地有embedding文件,就掉pytorch自带的方法,加载本地的文件
        if config.embedding_pretrained is not None:
            # freeze=False 不冻住,等下一起参与训练
            self.embedding = nn.Embedding.from_pretrained(config.embedding_pretrained, freeze=False)
        else:
            self.embedding = nn.Embedding(config.n_vocab, config.embed, padding_idx=config.n_vocab - 1)
        # lstm
        self.lstm = nn.LSTM(config.embed, config.hidden_size, config.num_layers,
                            # 是否是双向的,如果是双向的话,lstm会从前往后做一次传播,然后再从后往前做一次传播,这样本来是如果是输出128维的向量的话,就变成输出256维的向量,特征更多了
                            bidirectional=True
                            # pytorch都是batchfirst的
                            , batch_first=True
                            , dropout=config.dropout)
        # 最后用一个FC算出各个类别的值,config.hidden_size * 2是因为前面的bidirectional=True
        self.fc = nn.Linear(config.hidden_size * 2, config.num_classes)

    def forward(self, x):
        x, _ = x
        # 做完embedding后,输入就变成了[batch_size, seq_len, embeding]=[128, 32, 300]
        out = self.embedding(x)
        out, _ = self.lstm(out)
        # 因为是做分类任务,就之后句子序列最后一个词的输出结果,所以是[:,-1,:]
        out = self.fc(out[:, -1, :])
        return out
