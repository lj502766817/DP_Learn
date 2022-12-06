"""
自定义数据集的demo
"""
import torch
from torch_geometric.data import Data
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
from torch_geometric.data import InMemoryDataset
from tqdm import tqdm
from torch_geometric.nn.pool import TopKPooling
from torch_geometric.nn.conv import SAGEConv
from torch_geometric.nn.pool.glob import global_mean_pool as gap, global_max_pool as gmp
import torch.nn.functional as func

# 首先是一个超简单的例子,一个图,x是每个点的特征,y是标签
x_ = torch.tensor([[2, 1], [5, 6], [3, 7], [12, 0]], dtype=torch.float)
y_ = torch.tensor([0, 1, 0, 1], dtype=torch.float)
# 图里的边,每一对点中间没有排序
edge_index_ = torch.tensor([[0, 1, 2, 0, 3],  # 起始点
                            [1, 0, 1, 3, 2]], dtype=torch.long)  # 终止点
# 用geometric创建一个图的数据
data_ = Data(x=x_, y=y_, edge_index=edge_index_)
print(data_)  # Data(x=[4, 2], edge_index=[2, 5], y=[4]),图里有4个点,每个点2个特征,图中有5条边,标签有4个

# 用一个真实的用户点击浏览购买记录数据集
# 用户的点击记录,一个session_id对应一个人次的记录,意味着一个session_id就是一个图
df = pd.read_csv('yoochoose-clicks.dat', header=None)
df.columns = ['session_id', 'timestamp', 'item_id', 'category']
# 用户的购买记录,就是对应某个图的标签
buy_df = pd.read_csv('yoochoose-buys.dat', header=None)
buy_df.columns = ['session_id', 'timestamp', 'item_id', 'price', 'quantity']
# 把item_id转成数字编码
item_encoder = LabelEncoder()
df['item_id'] = item_encoder.fit_transform(df.item_id)
print(df.head())

# 拿一部分来玩
sampled_session_id = np.random.choice(df.session_id.unique(), 10000, replace=False)
df = df.loc[df.session_id.isin(sampled_session_id)]
df.nunique()
# 数据对应标签,有买还是没买
df['label'] = df.session_id.isin(buy_df.session_id)
print(df.head())


# 自定义数据集
class YooChooseBinaryDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        # transform就是数据增强，对每一个数据都执行
        super(YooChooseBinaryDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        # 检查self.raw_dir目录下是否存在raw_file_names()属性方法返回的每个文件
        # 如有文件不存在，则调用download()方法执行原始文件下载
        return []

    @property
    def processed_file_names(self):
        # 检查self.processed_dir目录下是否存在self.processed_file_names属性方法返回的所有文件，没有就会走process
        return ['yoochoose_click_binary_1M_sess.dataset']

    def download(self):
        pass

    def process(self):
        data_list = []

        # 根据session_id分组一个个的构建图
        grouped = df.groupby('session_id')
        for session_id, group in tqdm(grouped):
            # 先根据item_id从小到大把每个结点编号,因为geometric里对图中的每个结点都是0,1,2..这样编号的
            sess_item_id = LabelEncoder().fit_transform(group.item_id)
            # 重置df的index
            group = group.reset_index(drop=True)
            group['sess_item_id'] = sess_item_id
            # 把item_id作为每个点的特征
            node_features = group.loc[group.session_id == session_id, ['sess_item_id', 'item_id']].sort_values(
                'sess_item_id').item_id.drop_duplicates().values

            node_features = torch.LongTensor(node_features).unsqueeze(1)
            # 这里是做邻接矩阵,这个数据里的结点是按0号位->1号位,1号位->2号位,..这样链接的,所以起始点是从开头取到倒数第二个,结束点是从第二个取到末尾
            source_nodes = group.sess_item_id.values[:-1]
            target_nodes = group.sess_item_id.values[1:]

            edge_index = torch.tensor(np.array([source_nodes, target_nodes]), dtype=torch.long)
            x = node_features
            y = torch.FloatTensor([np.array(group.label.values, dtype=int)[0]])
            # 构造一个图
            data = Data(x=x, edge_index=edge_index, y=y)
            data_list.append(data)

        # 把图的list转成InMemoryDataset的格式,然后做个持久化
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


# 实例化数据集
dataset = YooChooseBinaryDataset(root='data/')

# 构建网络模型
embed_dim = 128


# 针对图进行分类任务
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        # 一个图卷积层
        # \mathbf{x}^{\prime}_i = \mathbf{W}_1 \mathbf{x}_i + \mathbf{W}_2 \cdot \mathrm{mean}_{j \in \mathcal{N(i)}} \mathbf{x}_j
        # 用当前点乘一个权重加上周围点的均值乘一个权重
        self.conv1 = SAGEConv(embed_dim, 128)
        # 池化操作选出前几名的结点,压缩图中的结点数和邻接矩阵大小
        # 用一个可学习的矩阵去和结点去计算他们各自的的分值,根据得分值选出前几名的结点
        self.pool1 = TopKPooling(128, ratio=0.8)
        self.conv2 = SAGEConv(128, 128)
        self.pool2 = TopKPooling(128, ratio=0.8)
        self.conv3 = SAGEConv(128, 128)
        self.pool3 = TopKPooling(128, ratio=0.8)
        # 类似NLP的词向量表,把item_id转成一个128维的向量
        self.item_embedding = torch.nn.Embedding(num_embeddings=df.item_id.max() + 10, embedding_dim=embed_dim)
        self.lin1 = torch.nn.Linear(128, 128)
        self.lin2 = torch.nn.Linear(128, 64)
        self.lin3 = torch.nn.Linear(64, 1)
        self.bn1 = torch.nn.BatchNorm1d(128)
        self.bn2 = torch.nn.BatchNorm1d(64)
        self.act1 = torch.nn.ReLU()
        self.act2 = torch.nn.ReLU()

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch  # x:n*1,其中每个图里点的个数是不同的
        # print(x)
        x = self.item_embedding(x)  # n*1*128 特征编码后的结果
        # print('item_embedding',x.shape)
        x = x.squeeze(1)  # n*128
        # print('squeeze',x.shape)
        x = func.relu(self.conv1(x, edge_index))  # n*128
        # print('conv1',x.shape)
        x, edge_index, _, batch, _, _ = self.pool1(x, edge_index, None, batch)  # pool之后得到 n*0.8个点
        # print('self.pool1',x.shape)
        # print('self.pool1',edge_index)
        # print('self.pool1',batch)
        # x1 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)
        x1 = gap(x, batch)
        # print('gmp',gmp(x, batch).shape) # batch*128
        # print('cat',x1.shape) # batch*256
        x = func.relu(self.conv2(x, edge_index))
        # print('conv2',x.shape)
        x, edge_index, _, batch, _, _ = self.pool2(x, edge_index, None, batch)
        # print('pool2',x.shape)
        # print('pool2',edge_index)
        # print('pool2',batch)
        # x2 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)
        x2 = gap(x, batch)
        # print('x2',x2.shape)
        x = func.relu(self.conv3(x, edge_index))
        # print('conv3',x.shape)
        x, edge_index, _, batch, _, _ = self.pool3(x, edge_index, None, batch)
        # print('pool3',x.shape)
        # x3 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)
        x3 = gap(x, batch)
        # print('x3',x3.shape)# batch * 256
        x = x1 + x2 + x3  # 获取不同尺度的全局特征

        x = self.lin1(x)
        # print('lin1',x.shape)
        x = self.act1(x)
        x = self.lin2(x)
        # print('lin2',x.shape)
        x = self.act2(x)
        x = func.dropout(x, p=0.5, training=self.training)

        x = torch.sigmoid(self.lin3(x)).squeeze(1)  # batch个结果
        # print('sigmoid',x.shape)
        return x
