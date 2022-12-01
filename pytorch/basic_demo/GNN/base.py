"""
GCN一些基本操作
"""
import torch
import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.datasets import KarateClub
from torch_geometric.utils import to_networkx
from torch.nn import Linear
from torch_geometric.nn import GCNConv


# 画图展示的一些函数
def visualize_graph(G, color):
    plt.figure(figsize=(7, 7))
    plt.xticks([])
    plt.yticks([])
    nx.draw_networkx(G, pos=nx.spring_layout(G, seed=42), with_labels=False,
                     node_color=color, cmap="Set2")
    plt.show()


def visualize_embedding(h, color, epoch=None, loss=None):
    plt.figure(figsize=(7, 7))
    plt.xticks([])
    plt.yticks([])
    h = h.detach().cpu().numpy()
    plt.scatter(h[:, 0], h[:, 1], s=140, c=color, cmap="Set2")
    if epoch is not None and loss is not None:
        plt.xlabel(f'Epoch: {epoch}, Loss: {loss.item():.4f}', fontsize=16)
    plt.show()


# geometric内置的一份数据集
dataset = KarateClub()
print(f'Dataset: {dataset}:')
print('======================')
print(f'Number of graphs: {len(dataset)}')
print(f'Number of features: {dataset.num_features}')
print(f'Number of classes: {dataset.num_classes}')
# Data(x=[34, 34], edge_index=[2, 156], y=[34], train_mask=[34])
# 图里有34个点,每个点是34个特征,图里面有156条边,edge_index表示边从哪个点到哪个点的边,所以第一维是2.train_mask表示有的点是没有标签的,是个半监督的
data = dataset[0]
print(data)

# 看看这个图
G = to_networkx(data, to_undirected=True)
visualize_graph(G, color=data.y)


# 用这个数据集做一个基本的分类任务
class GCN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(1234)
        # 只需定义好输入特征和输出特征即可
        self.conv1 = GCNConv(dataset.num_features, 4)
        self.conv2 = GCNConv(4, 4)
        self.conv3 = GCNConv(4, 2)
        self.classifier = Linear(2, dataset.num_classes)

    def forward(self, x, edge_index):
        # 输入特征与邻接矩阵,数据格式是和数据集对应的
        h = self.conv1(x, edge_index)
        h = h.tanh()
        h = self.conv2(h, edge_index)
        h = h.tanh()
        h = self.conv3(h, edge_index)
        h = h.tanh()

        # 分类层
        out = self.classifier(h)

        return out, h  # h是2维的中间量,方便画图


model = GCN()
print(model)

# 看看原始数据分布
model = GCN()
_, h = model(data.x, data.edge_index)
print(f'Embedding shape: {list(h.shape)}')
visualize_embedding(h, color=data.y)

# 开始训练,损失函数和优化器都是常规套路
model = GCN()
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)


def train(data):
    optimizer.zero_grad()
    out, h = model(data.x, data.edge_index)
    loss = criterion(out[data.train_mask], data.y[data.train_mask])  # 半监督任务的提现
    loss.backward()
    optimizer.step()
    return loss, h


for epoch in range(401):
    loss, h = train(data)
    if epoch % 80 == 0:  # 80次迭代看一下
        visualize_embedding(h, color=data.y, epoch=epoch, loss=loss)
# 随着迭代进行,在2维平面上,逐渐有聚类的效果
