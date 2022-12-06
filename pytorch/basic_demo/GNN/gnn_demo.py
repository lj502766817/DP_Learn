"""
gnn的demo
"""
from torch_geometric.datasets import Planetoid  # 下载数据集用的
from torch_geometric.transforms import NormalizeFeatures
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE  # 做降维,好画图
import torch
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


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
# 传统的全连接
class MLP(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        torch.manual_seed(12345)
        self.lin1 = Linear(dataset.num_features, hidden_channels)
        self.lin2 = Linear(hidden_channels, dataset.num_classes)

    def forward(self, x):
        x = self.lin1(x)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        return x


model = MLP(hidden_channels=16)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)


def train():
    model.train()
    optimizer.zero_grad()
    out = model(data.x)
    loss = criterion(out[data.train_mask],
                     data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss


def test():
    model.eval()
    out = model(data.x)
    pred = out.argmax(dim=1)
    test_correct = pred[data.test_mask] == data.y[data.test_mask]
    test_acc = int(test_correct.sum()) / int(data.test_mask.sum())
    return test_acc


for epoch in range(1, 201):
    loss = train()
    if epoch % 10 == 0:
        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')  # 200次迭代后损失从1.8893降低到了0.3810
test_acc = test()
print(f'Test Accuracy: {test_acc:.4f}')  # 准确度是0.59


# 换成GNN模型
class GCN(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        torch.manual_seed(1234567)
        self.conv1 = GCNConv(dataset.num_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, dataset.num_classes)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return x


# 先看看初始情况
model = GCN(hidden_channels=16)
model.eval()
out = model(data.x, data.edge_index)
visualize(out, color=data.y)

# 再做做训练
model = GCN(hidden_channels=16)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
criterion = torch.nn.CrossEntropyLoss()


def train():
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = criterion(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss


def test():
    model.eval()
    out = model(data.x, data.edge_index)
    pred = out.argmax(dim=1)
    test_correct = pred[data.test_mask] == data.y[data.test_mask]
    test_acc = int(test_correct.sum()) / int(data.test_mask.sum())
    return test_acc


for epoch in range(1, 201):
    loss = train()
    if epoch % 10 == 0:
        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')  # 同样是200次迭代,损失从1.8685降到0.3045
test_acc = test()
print(f'Test Accuracy: {test_acc:.4f}')  # 准确率到了0.8020

# 看看训练后的结果
model.eval()
out = model(data.x, data.edge_index)
visualize(out, color=data.y)  # 对比有明显的聚类结果
