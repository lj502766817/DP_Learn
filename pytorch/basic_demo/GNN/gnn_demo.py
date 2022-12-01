"""
gnn的点分类任务
"""
from torch_geometric.datasets import Planetoid  # 下载数据集用的
from torch_geometric.transforms import NormalizeFeatures
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE  # 做降维,好画图


# 画图的函数
def visualize(h, color):
    z = TSNE(n_components=2).fit_transform(h.detach().cpu().numpy())

    plt.figure(figsize=(10, 10))
    plt.xticks([])
    plt.yticks([])

    plt.scatter(z[:, 0], z[:, 1], s=70, c=color, cmap="Set2")
    plt.show()


# 下载要能上github才行,transform是做预处理操作
dataset = Planetoid(root='../data/Planetoid', name='Cora', transform=NormalizeFeatures())

# 一个论文引用的数据集,这个数据里面是1个图,每个点是1433个特征,7个类别
print(f'Dataset: {dataset}:')
print('======================')
print(f'Number of graphs: {len(dataset)}')
print(f'Number of features: {dataset.num_features}')
print(f'Number of classes: {dataset.num_classes}')

data = dataset[0]
print(data)
print('===========================================================================================================')
print(f'Number of nodes: {data.num_nodes}')  # 2708个点
print(f'Number of edges: {data.num_edges}')  # 10556个边
print(f'Average node degree: {data.num_edges / data.num_nodes:.2f}')  # 一个点平均3.9个度
print(f'Number of training nodes: {data.train_mask.sum()}')  # 有标签的点是140个
print(f'Training node label rate: {int(data.train_mask.sum()) / data.num_nodes:.2f}')
print(f'Has isolated nodes: {data.has_isolated_nodes()}')
print(f'Has self-loops: {data.has_self_loops()}')
print(f'Is undirected: {data.is_undirected()}')

# 传统mlp和gnn的对比
