### Transformer笔记

#### 与传统的rnn对比

传统rnn的问题

* 传统的rnn网络是通过固定的词向量语料库来训练一个模型,那么模型训练好了之后,每个输入的token(将每个词转换成词向量,这个向量就叫token)
  就可以说是定死了的,即每个词在这个模型中他的特征就定死了.但是在实际语言环境中缺不是这样,相同的一个词,在不同的语境下,表示的意义可以是完全不一样的.所以传统的rnn网络做出的模型,效果不是很理想.
* 传统的rnn网络是一个串行的网络,串行就意味着模型的训练周期是很长的,并且由于串行的时间长,那么也不能进行多层的堆叠,因为这样周期就更久了

但是transformer解决了这些问题

* transformer使用了attention机制,对于每个词,transformer并没有用一个固定的token来表示它,而是根据上下文去产生一个综合的token去表示,这样就避免了传统rnn不能适应各种语境的问题
* transformer的self-attention是并行计算的,输入和输出都是相同的,都是同时被计算出来的,所以在时间上是大大加快的

#### self-attention机制

因为有语境这个条件了,那么一句话中的每个词对应的token现在不能单独考虑这个词本身了,每个token的组成都需要结合上下文的的各个词.

假设现在有一段话:$我吃西瓜$,通过语料表将它转换成token之后是: $[A,B,C,D]$ ,在传统的rnn中所有的token就这样定死了,这样就丢掉了语境.但是在transformer里,会将这个token进行进一步的加工(
encoder): $A^\prime=w_1A+w_2B+w_3C+w_4D$ ,得到新的token: $[A^\prime,B^\prime,C^\prime,D^\prime]$
,这个新的token就具有了上下文的信息.现在的问题就变成了这个 $w$ ,怎么得到?

在transformer中,我们设置三个辅助向量Q(query),K(keys),V(value)
来做这个事,Q向量表示查询向量,是一个token向每个token去查询关联时用到,K向量表示应答向量,是当token被查询时,来做回应时用到,V向量表示真实特征向量,是每个token的真实特征值.那么在计算 $w_2$
的时候,我们就可以用向量 $q_A$ 与向量 $k_B$
做内积,因为内积是可以算出两个向量的相关程度的,这个时候计算出的内积值就是A和B的关联程度了,依次类推,我们可以计算A与A自身,与B,与C,与D的关联程度.又因为内积的值很可能是大于1的数值,我们就需要把这些值做成权重值,这个时候可以用softmax函数来进行转换得到对应的的
$[w_1,w_2,w_3,w_4]$ ,然后我们用对应的权重值乘上对应token的各自真实值V,再相加,就得到了转换后的具有上下文信息的新token:

$$ \vec{Z}=softmax({{\vec Q \cdot \vec K}\over \sqrt{d_k}})\cdot \vec{V} $$

这里的 $d_k$ 表示对向量的维度,因为Q,K,V是我们自己定的,这里是为了避免因为向量的维度大小而对结果造成很大的影响,取根号是实验得出的结果

#### muti-headed机制(多头机制)

我们可以使用多次自注意力机制,得到多个Z向量,然后用一个FC来做降维,得到我们需要大小的结果,类似于卷积核的概念,用多个卷积核来提特征,最后综合到一起.

实际使用过程中,假设我们的token是512维的,我们需要做8头的话,我们可以每个头做64维,最后把8头拼在一起,最后就用FC去做综合

#### 多层堆叠

不同于传统的rnn是串行的,transformer是并行计算的,在时间效率上有很大提升,我们就可以使每个词的token在做encoder时输入与输出的维度一样,然后就可以在第一层的encoder上再加一层encoder,进一步的提取特征.

#### 位置编码

除了考虑上下文以外,位置信息也有重要的影响,例如:我打你,你打我,这两句话虽然有相同的字,但是表达的意思完全不同.所以transformer在做encoder的时候还会把位置信息加进去,例如:
第一个token加上位置信息1,第二个token加上位置信息2...,一般我们会想用one-hot编码来表达位置信息,像1000,0100,这种.不过实际情况下是用正余弦编码器来实现.

##### 正余弦位置编码

论文中的位置编码是一个 $d$ (这个 $d$ 需要能被2整除)维的向量.这个向量可以表示一个位置的特征,为每个单词表示它在句子中的位置.定义位置编码的函数:

$$
\vec{p_t}^{(i)}=f(t)^{(i)}=
\begin{cases}
sin(\omega_k\cdot t),\quad i=2k\\
cos(\omega_k\cdot t), \quad i=2k+1
\end{cases}
$$

其中 $\omega_k={1 \over 10000^{2k \over d}}$ , $i$ 表示第 $d$ 维向量中第几个.那么最后这个位置编码向量就是这样的形式:

$$
\vec{p_t}=
\left[
\begin{matrix}
sin(\omega_1\cdot t)  \\
cos(\omega_1\cdot t)  \\
sin(\omega_2\cdot t)  \\
cos(\omega_2\cdot t)  \\
\vdots  \\
sin(\omega_{d \over 2}\cdot t)  \\
cos(\omega_{d \over 2}\cdot t)  \\
\end{matrix}
\right]
$$

并且这个正余弦编码也能表达各个位置的相对关系:

$$
M \cdot
\left[
\begin{matrix}
sin(\omega_k\cdot t)  \\
cos(\omega_k\cdot t)  \\
\end{matrix}
\right]
=
\left[
\begin{matrix}
sin(\omega_k\cdot (t+\phi))  \\
cos(\omega_k\cdot (t+\phi))  \\
\end{matrix}
\right]
$$

其中 $M$ 是可求的,并且与 $t$ 无关.



#### 残差连接和归一化

在每一个block做完之后,将原始的输入数据与做完encoder之后的结果相加,进行残差链接,类似于resnet.并且将做完残连接后的数据做一个Layer Normalization(在每一层对单个样本的所有神经元节点进行规范化)

#### attention机制(decoder)

一个input经过encoder之后会得到一个token的序列,到实际做任务的时候,可能需要做一个attention(decoder)
.这里用机器翻译做例子说明.做翻译时,实际是一个词一个词的翻译,这时,我们首先是也对output的第一个词做成token,此时因为是翻译任务,我们就认为后面的词是不知道的,需要对后面的词做一个mask,然后,我们就用第一个词的token的Q向量一个一个的去查询input序列的各个K向量,得到权重然后通过V向量计算output第一个词经过attention机制后的token,得到第一个词的实际token后,然后得到第二个词,然后第二个词不仅需要跟input做attention,还要与第一个词做self-attention.这个就是decoder.

decoder之后的结果我们再做linear和softmax,就能得到一个预测的机器翻译的序列了

![transformer](https://user-images.githubusercontent.com/28779173/190123789-dc4335e2-ba6f-4252-abc8-ab962179a89d.jpg)
