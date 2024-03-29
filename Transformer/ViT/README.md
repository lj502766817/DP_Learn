### ViT(Vision Transformer的)
在transformer里,输入的数据需要做成一个token的序列.
那么,针对图像数据,我们可以先把图像数据分成一个一个的patch,然后对每个patch加上位置编码,
然后,一个图像数据就变成了一个token的序列.此时这个序列就可以放到transformer里进行处理了.

例如,针对一个分类的任务,我们可以在处理这个图像数据序列的时候,附加一个cls向量,一起放到
transformer里进行处理.这样,这个cls向量就有个这个图像数据的全部特征,后续就是正常的分类数据
处理了.

因此,精髓就是如何将一个图像数据处理成一个序列的token.transformer在视觉
领域中的优势就是感受野的优势,传统的CNN网络需要不断地做卷积核池化来扩大感受野,但是在transformer
的架构下,第一层就可以看到整个图像的数据,得到的信息更加的丰富

#### 位置编码说明
位置编码在处理图像数据的时候是必须的,因为要构成一个图像数据序列.但是如何编码,通过实验发现
,2D(行,列数据)的编码和1D(顺序数据)的编码对结果没太大影响.

![ViT](https://user-images.githubusercontent.com/28779173/188341225-608836f5-c439-4c88-b71f-b27c894f1e5b.png)

### TNT(Transformer in Transformer)
前面说的vit是针对图像数据做的一个一个patch进行建模,但是patch里面更小的细节是忽略掉了的,
所以出现了TNT,TNT的处理方式是图像数据分出的一个个patch,外层的transformer还是像前面一样
正常的处理.

但是针对一个个patch本身,里面的数据,我们就将patch本身再重组成一个超像素的序列,然后对这个序列
继续做transformer.例如:假如,一个patch本身是16*16大小的数据,做成超像素的时候,可以把超像素
序列做成4*4的,就是本身一个16*16的patch,给他继续分化成4*4的一个更小的patch序列,对这个超像素
序列继续加位置编码,做transformer,最后通过FC把输出向量的大小做成和外层的transformer的输入
向量大小一样,最后把两个向量相加,再一起做外层的transformer.

