### Informer原理说明

Informer的主要目的是为了解决一个长的时间序列预测问题.一些经典的时间序列算法,如:

* Prophet:适合预测趋势,但是预测数值不够精准
* Arima:短序列的预测比较精准,但是不能做趋势的预测
* LSTM:在长序列的情况下,由于是串行,运算速度很慢,并且效果也不够理想

这些算法在长时间序列上都不适用,于是就在Transformer架构的基础上,提出了Informer.

传统Transformer优劣:

* Transformer里可以对每个token做并行计算,比串行快很多
* attention机制是对每个token做,长序列的情况下计算量太大
* 传统Transformer的decoder层预测是一个一个的输出,输出的能力不够理想

因此Informer里就需要解决:**Attention的快速计算**,**Decoder需要一次性输出全部的预测**,**Encoder的堆叠也要做的够快**,这三个问题

#### Informer中Attention的计算

在一个长序列中,并不是每个位置的attention都是那么重要的,通过Informer论文可以知道,实际上有比较大作用的attention只是占一小部分.这一部分位置的Q向量和其他位置的K向量的乘积结果的起伏是很大的,也从侧面说明了这部分位置的作用大,但是大部分的其他位置的Q向量的乘积结果都是很平稳的,就表名这部分位置没什么特点,对结果的影响没那么大.

那么如何去定义一个Q是不是一个有作用的Q就很关键了,就可以用每个Q和均匀分布的差异来表示,差异越大说明这个Q越有价值:

$$
M(q_i,K)=\ln{\sum^{L_K}_{j=1}}e^{{q_ik_j^T} \over \sqrt{d}}-{1\over L_K}\sum^{L_K}_{j=1}{{q_ik_j^T} \over \sqrt{d}}
$$

##### ProbAttention的计算

传统的Transformer里是计算self-attention,而在Informer里为了长序列的情况下Attention的快速计算,设计了ProbAttention来代替传统的self-attention.

ProbAttention的第一步就是要找出有价值的Q.假设序列的长度是96,那么首先就随机的选出25个K,因为如果这个Q是重要的Q,那么他和这个随机选出的25个K的计算结果结果起伏很大的.反之,如果这个Q不重要,那么这25个计算结果肯定也是比较均匀的.

那么现在对每个Q就有了25个得分值了(与25个K计算的结果),在论文的中,为了进一步加速计算,对Q的得分值是直接用最大值和均匀分布的差异来表示每个Q的重要性得分(因为通常情况下,重要的Q不是突变的,一个重要位置的周围通常也是相对重要的位置):

$$
\overline M(q_i,K)=\max_j\{{{q_ik_j^T} \over \sqrt{d}}\}-{1\over L_K}\sum^{L_K}_{j=1}{{q_ik_j^T} \over \sqrt{d}}
$$

这里已经选出了25个重要的Q,就可以得到这25个位置的attention值,但是一个序列是96个长度,那么其他位置的attention值是不知道的,在论文中,其他位置的attention值是用的V的均值来替代的.最后就是选出来的25个Q是会更新的,其他的都是均值向量结果

##### Self-Attention Distilling

在Transformer的架构中,self-attention不是做一次的,而是要进行多层的堆叠.在Informer中也同样会这么做,但是和传统的Transformer不一样的是,在Informer中,做完一轮self-attention(ProbAttention)后的结果会经过一个Conv1d(FC也同理)来进行下采样,这样下一轮的self-attention的输入就是变成48,而重要的Q也只要20个了,这种蒸馏的操作进一步的加速了长序列的计算

并且不仅仅是下采样操作,在Distilling中也加入了Stamp(各种时间特征编码与位置编码)也融合到整体的特征中

![Informer Distilling](https://user-images.githubusercontent.com/28779173/190934881-f65799cb-e386-45a3-86f4-e856e428c85e.png)

ProbAttention和Self-Attention Distilling合起来就构成了Informer里Encoder的架构

#### Informer中Decoder输出

传统的Transformer里,Decoder的输出是一个一个的,因为在传统的Transformer里,decoder的第二个输出是要和第一个输出去计算attention的.

在Informer中是可以直接一次性的输出一个序列.他的做法是用一段真实值去做辅助.例如我们想得到20-30号的输出结果,我们会在前面预先给出一个10-20号的结果来辅助

还是以前面96长度的序列为基本例子,假如想要预测的长度为24,那么decoder的输入其实并不像传统的transformer就是24,而是输入一个长度为72的序列,这前面的48个序列是已知的真实值,目的就是用这48个真实值来辅助decoder,做一次性的输出.

在decoder中,第一步还是自身要做ProbAttention,但是这个时候需要加上mask,就是前面位置不能和后面位置做attention(不能预知),然后第二部就是正常的和encoder的输出做attention

#### 总结

Informer相对于传统的Transformer就是对编码器和解码器的优化,使其速度更块,解决了长序列的问题

![Informer架构](https://user-images.githubusercontent.com/28779173/190935398-5ebd4a95-93b5-476d-a231-93f9cca5f711.png)