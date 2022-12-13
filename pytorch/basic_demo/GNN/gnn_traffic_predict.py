"""
交通速度预测
"""
from torch_geometric_temporal.dataset import METRLADatasetLoader
import seaborn as sns
import matplotlib.pyplot as plt
from torch_geometric_temporal.signal import temporal_signal_split
import torch
import torch.nn.functional as func
from torch_geometric_temporal.nn.recurrent import A3TGCN  # 带注意力机制的时序图卷积

loader = METRLADatasetLoader()
# 输入12个样本点 预测未来12个点的值
dataset = loader.get_dataset(num_timesteps_in=12, num_timesteps_out=12)
print("样本个数: ", dataset.snapshot_count)
# 一个图里207个点(传感器),每个点是有12个时间点的速度值(两个特征:速度,时间),标签是12个时间点的速度值
print(next(iter(dataset)))

# 看看数据分布,1号传感器的24小时记录的速度值
sensor_number = 1
hours = 24
sensor_labels = [bucket.y[sensor_number][0].item() for bucket in list(dataset)[:hours]]
sns.lineplot(data=sensor_labels)
plt.show()

# 切分数据集
train_dataset, test_dataset = temporal_signal_split(dataset, train_ratio=0.8)
print("训练集: ", len(list(train_dataset)))
print("测试集: ", len(list(test_dataset)))


# 模型定义
class TemporalGNN(torch.nn.Module):
    def __init__(self, node_features, periods):
        super(TemporalGNN, self).__init__()
        # Attention Temporal Graph Convolutional Cell
        self.tgnn = A3TGCN(in_channels=node_features,
                           out_channels=32,
                           periods=periods)
        # 预测未来的12个值
        self.linear = torch.nn.Linear(32, periods)

    def forward(self, x, edge_index):
        """
        x = Node features for T time steps
        edge_index = Graph edge indices
        """
        h = self.tgnn(x, edge_index)
        h = func.relu(h)
        h = self.linear(h)
        return h


device = torch.device('cpu')
# 2000个batch做一次反向传播
subset = 2000

# 模型和优化器
model = TemporalGNN(node_features=2, periods=12).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
# 训练
model.train()
print("Running training...")
for epoch in range(20):
    loss = 0
    step = 0
    for snapshot in train_dataset:
        snapshot = snapshot.to(device)
        y_hat = model(snapshot.x, snapshot.edge_index)
        # MSE
        loss = loss + torch.mean((y_hat - snapshot.y) ** 2)
        step += 1
        if step > subset:
            break

    loss = loss / (step + 1)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    print("Epoch {} train MSE: {:.4f}".format(epoch, loss.item()))
# 验证
model.eval()
loss = 0
step = 0
horizon = 288

predictions = []
labels = []
for snapshot in test_dataset:
    snapshot = snapshot.to(device)
    y_hat = model(snapshot.x, snapshot.edge_index)
    loss = loss + torch.mean((y_hat - snapshot.y) ** 2)
    labels.append(snapshot.y)
    predictions.append(y_hat)
    step += 1
    if step > horizon:
        break

loss = loss / (step + 1)
loss = loss.item()
print("Test MSE: {:.4f}".format(loss))
