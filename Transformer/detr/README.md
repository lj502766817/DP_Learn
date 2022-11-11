### Detr说明

Detr属于是在one-stage的目标检测开辟了一个新的领域.之前的fast-rcnn,yolo这些都是围绕这anchor来的.并且最后都是需要通过NMS来做过滤.这里的anchor就属于是一些人为的因素在里面,并且NMS也是一个比较耗时的操作.而基于transformer架构的Detr是属于一个完全排除了人为因素,不使用anchor,直接给出检测框的一个网络.

Detr的基本思路就是:首先使用CNN来将图片切成一个个patch,然后经过transformer的encode(这一步和ViT基本是一样的),最后通过decoder直接得到100(这个100是直接写死的)个检测的坐标框.

#### 整体网络结构

![Detr网络结构](https://user-images.githubusercontent.com/28779173/201287246-45349e08-eb03-46f5-a997-eb02fdac88e0.png)

主要的目的就是得到最终输出的object query向量,此时的query向量经过Transformer的encoder与decoder已经学会了在原始特征中获得物体的位置,然后通过query向量经过FFN来得到预测的目标框和对应的分类值.

#### 一些细节

* 与NLP中的Transformer不同的是,这里的decoder解码query向量是可以并行处理的.
* 解码器的query向量在初始化的时候是直接使用位置编码做初始化,然后一层一层堆叠来学习输入的特征
* 输出的时候,因为默认的预测框是固定100个,但是实际的GT数量不一定.这个时候是使用匈牙利算法,按照最小的loss,来得到最佳的匹配.
