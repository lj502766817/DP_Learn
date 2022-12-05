"""
自定义数据集的demo
"""
import torch
from torch_geometric.data import Data

# 首先是一个超简单的例子,一个图,x是每个点的特征,y是标签
x = torch.tensor([[2, 1], [5, 6], [3, 7], [12, 0]], dtype=torch.float)
y = torch.tensor([0, 1, 0, 1], dtype=torch.float)
# 图里的边,每一对点中间没有排序
edge_index = torch.tensor([[0, 1, 2, 0, 3],  # 起始点
                           [1, 0, 1, 3, 2]], dtype=torch.long)  # 终止点
# 用geometric创建一个图的数据
data = Data(x=x, y=y, edge_index=edge_index)
print(data)  # Data(x=[4, 2], edge_index=[2, 5], y=[4]),图里有4个点,每个点2个特征,图中有5条边,标签有4个
