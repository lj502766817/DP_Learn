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

##### GNN的注意力机制

GNN的注意力机制是作用在结点上的,无注意力机制的情况下,一个结点的更新是基于邻接结点取平均.

但是在注意力机制的作用下,这个更新并不是平均的,而是当前节点与邻接结点的关系权重.这个关系权重在结点的更新中就体现在邻接矩阵的值不再是1了,而是一个注意力的加权值.

###### 加权值的计算

这个权重计算方式有很多,举例一种.参考经典论文:[Graph Attention Networks](https://arxiv.org/abs/1710.10903)

假设两个结点A和B是邻接的,它们各自的特征向量是 $h_a$ , $h_b$ .乘上权重之后就是 $Wh_a$ , $Wh_b$ .然后把这两个矩阵拼起来,再乘上一个计算注意力的权重就能得到它们两个之间的一个注意力权重值 $e_{ab}$ ,再过一个 $softmax$ 就是对应邻接矩阵的加权值了.

对应结点的特征值更新就是在这个计算过加权之后的邻接矩阵上做.

