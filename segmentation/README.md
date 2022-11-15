### 分割任务

* 语义分割

  逐像素的进行分类,把每个像素分成人,动物,背景等等.语义分割只进行类别的分割,类别内的不做区分

* 实例分割

  与语义分割不同的是还需要分割出类别内的每个个体

#### 损失函数

由于实际是一个像素点的分类,因此损失函数基本是一个交叉熵损失函数了.

不过由于还需要考虑样本的均衡问题,因此还会附加一个权重值 $posWeight={numNeg \over numPos}$ ,那么附加权重后的交叉熵损失函数就是:

$$
loss=posWeight \times y_{true}log(y_{pred}) - (1-y_{true})log(1-y_{pred})
$$

##### Focal Loss

除了考虑样本均衡的权重.对样本的难易程度也需要加上考虑,简单样本给一个小一点权重,困难样本给一个大一点的权重,然后再把权重结合进来:

$$
loss=-\alpha(1-y_{pred})^\gamma \times y_{true}log(y_{pred}) -(1-\alpha)y_{pred}^\gamma \times (1-y_{true})log(1-y_{pred})
$$

其中 $\gamma$ 通常取2.

##### MIOU

这里的IOU和目标检测的IOU是一样的,只不过分割任务里会取个平均就是MIOU了
