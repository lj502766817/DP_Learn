### SwinTransformer原理说明

swin-transformer是在ViT的基础上进行的变种,相较与原始的ViT,swin-transformer使用了窗口和分层的模式来进行优化.在原始的ViT上,因为一个图像的像素点可能很多,这样想要提取更丰富的特征就会构建一个很长的序列.这样做attention的效率就很慢.但是在swin-transformer中,做了一个类似CNN网络的事,在每一层对特征图使用一个个窗口去做分解,对每个窗口来做attention,这样针对窗口的attention效率就更高.并且基于窗口的分层,也能做到类似CNN网络里感受野增强的效果

![swin-transformer](https://user-images.githubusercontent.com/28779173/189275452-fcdabda6-b939-41f4-b8b8-2b90d491c957.jpg)

#### W-MSA和SW-MS

在一个swin-transformer的block里是有一组两个小的block的,分别是W-MSA(窗口化多头自注意力机制)和SW-MSA(滑动窗口化多头自注意力机制),这个两个是固定一组并且是先W-MSA然后在SW-MSA

- W-MSA

  在一个原始的图像数据通过patch和embedding之后,我们现在得到的是这个原始图像数据特征图,这一步和正常的ViT是一样的.然后就是W-MSA的操作,在这个block里,我们先将这个特征图划分成一个个的窗口(window_partition),然后对每个窗口去做自己的自注意力机制,而不是像ViT那样整个的去做自注意力.做完窗口的self-attention后,我们就会把这些窗口还原成特征图,但是此时的特征图已经不是原来的那个特征图了,而是经过了self-attention之后的特征图了

- SW-MSA

  上一步我我们得到了基于窗口做了self-attention之后的特征图,但由于这个self-attention是在一个个窗口上做的,它的attention机制是都是窗口内部的关系,并没有相互之间的关系,这就会让模型局限在自己的那一小块窗口上.因此,我们需要让窗口做个偏移(shift).所谓偏移就是让窗口计算的像素点移位,做到一个滑动窗口的样子.

  ![滑动窗口](https://user-images.githubusercontent.com/28779173/189150373-63fd1733-006b-4d76-82bd-42586fb942ee.png)

  如图所示,窗口往右下滑动了几个像素点.这样滑动后的窗口(layer l+1)相对于滑动前的窗口(layer l)是覆盖了滑动前的边界了,这样就使之前相互独立的窗口产生了联系,但是滑动后,需要计算的注意力机制的窗口从4个变成了9个(w\*w变成了(w+1)\*(w+1)),因为外层被割裂开的窗口因为他们本身是不相连的,所以需要单独计算注意力,只有中间的那个整体因为滑动前后都是相连的,可以一起计算.这样就增加了计算的要求

















![整体架构](https://user-images.githubusercontent.com/28779173/189150418-1a4e36ca-eb98-44f4-a6e8-ac0027466dde.png)


