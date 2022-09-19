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

在一个长序列中,并不是每个位置的attention都是那么重要的,通过Informer论文可以看到,实际上有比价大作用的attention只是占一小部分.这一部分的
![Informer Distilling](https://user-images.githubusercontent.com/28779173/190934881-f65799cb-e386-45a3-86f4-e856e428c85e.png)
