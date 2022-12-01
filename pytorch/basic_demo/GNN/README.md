### 图神经网络

工具包:pytorch_geometric:https://github.com/pyg-team/pytorch_geometric

GNN模型主要解决问题是用来处理不规则的数据结构,像图像或者文本都可以做成规则的数据结构,但是一些医学,化学,或者一些工程上的那些分子结构,蛋白质啊,芯片结构啊的,就不好做成规则的数据了.

GNN模型的迭代更新,主要是基于图中的每个结点以及邻居节点的的信息来完成的.

图卷积一个层的定义:

$$
\mathbf{x}_v^{(\ell + 1)} = \mathbf{W}^{(\ell + 1)} \sum_{w \in \mathcal{N}(v) \, \cup \, \{ v \}} \frac{1}{c_{w,v}}
\cdot \mathbf{x}_w^{(\ell)}
$$

具体看论文:[Kipf et al. (2017)](https://arxiv.org/abs/1609.02907),或者geometric的文档:[GCNConv](https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.conv.GCNConv)

