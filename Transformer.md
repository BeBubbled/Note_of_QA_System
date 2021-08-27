[TOC]

# 背景知识

## encoder回顾/区分

[Difference between auto-encoder and encoder-decoder](https://datascience.stackexchange.com/questions/53979/what-is-the-difference-between-an-autoencoder-and-an-encoder-decoder)

auto-encoder是一种特殊的encoder-decoder, 它的输入与输出内容长得一样(或者说decoder尽可能还原encoder压缩出来的$\C$, 可以认为auto-encoder是一种有损压缩算法), $Auto-encoder\in Encoder-Decoder$

auto-encoder用途: 把中间的 hidden layer抽出来玩 [cite from](https://www.zhihu.com/question/338890317/answer/798264440)

encoder-decoder是一系列模型的统称, 是一种设计模式, 包含encoder与decoder这两个部分, encoder负责把属于映射至latent-space([在latent -space中, 相似的数据在latent空间中会得到相近的向量](https://www.quora.com/What-is-the-meaning-of-latent-space))中拿到向量$\C$, decoder负责解析$\C$,转换为其他东西

seq2seq的严格定义: seq2seq 是一个 **Encoder–Decoder 结构**的网络，它的输入是一个序列，输出也是一个序列， Encoder 中将一个可变长度的信号序列变为固定长度的向量表达，Decoder 将这个固定长度的向量变成可变长度的目标的信号序列. [cite from](https://www.jianshu.com/p/1d3de928f40c)

$seq2seq\in encoder-decoder$

$seq2seq2与auto-encoder$的区别: seq2seq的输入与输出内容不一样, seq2seq不符合auto-encoder的定义

**Seq2Seq和 Encoder-Decoder的关系** [cite from](https://easyai.tech/ai-definition/encoder-decoder-seq2seq/#Seq2Seq)

Seq2Seq（强调目的）不特指具体方法，满足「输入序列、输出序列」的目的，都可以统称为 Seq2Seq 模型。

而 Seq2Seq 使用的具体方法基本都属于Encoder-Decoder 模型（强调方法）的范畴。

总结一下的话：

- Seq2Seq 属于 Encoder-Decoder 的大范畴
- Seq2Seq 更强调目的，Encoder-Decoder 更强调方法

## RNN 变长序列

1. 为什么需要处理变长序? 
   
   因为神经网络需要输入内容具有较为一致的大小与格式

2. 变长序列的处理方法称为padding[^padding]

# Conditional Generation by RNN&Attention

[李宏毅P51](https://www.bilibili.com/video/BV1JE411g7XF?p=51)

* character/words
  * 英文中character组成word, word之间以空格相互分隔,组成句子
  * 中文: e.g. "葡萄" 是一个word, "葡"与"萄"是characters

# Transformer-> Bert

transfoermet: seq2seq model with self-attention

### Attention的计算原理

![2019-11-13-3step](Transformer/2019-11-13-3step.png)

key-value, query与key做F(Q,K) 匹配度计算, 拿到$s_{i}$, 最后得到$Attention_value=\sum_\limits{i}s_{i}\cdot value_{i}$

#### 在未应用attention时的encoder-decoder框架

![img](Transformer/0.png)

这是对于decoder f 来说, $Y_{i}$的产生顺序:
$$
\begin{array}{l}
\mathbf{y}_{\mathbf{1}}=\mathbf{f}(\mathbf{C}) \\
\mathbf{y}_{2}=\mathbf{f}\left(\mathbf{C}, \mathbf{y}_{1}\right) \\
\mathbf{y}_{3}=\mathbf{f}\left(\mathbf{C}, \mathbf{y}_{1}, \mathbf{y}_{2}\right)
\end{array}
$$

##### 在应用了attention后的encoder-decoder框架

![img](Transformer/0-20210314215842793.png)

现在
$$
\begin{array}{l}
\mathbf{y}_{\mathbf{1}}=\mathbf{f}(\mathbf{C}_{1}) \\
\mathbf{y}_{2}=\mathbf{f}\left(\mathbf{C}_{2}, \mathbf{y}_{1}\right) \\
\mathbf{y}_{3}=\mathbf{f}\left(\mathbf{C}_{3}, \mathbf{y}_{1}, \mathbf{y}_{2}\right)
\end{array}
$$
假设decoder部分$Y_{i}$部分具体展开如图

| -                                                                   | -                                                                   |
| ------------------------------------------------------------------- | ------------------------------------------------------------------- |
| ![image-20210314220632382](Transformer/image-20210314220632382.png) | ![image-20210314220641471](Transformer/image-20210314220641471.png) |

假设现在i==1

所以我们手里有$decoder隐藏层H_{0}, 权重计算函数s(a,b),encoder的每一个隐藏层h_{i}$

所以我们对$H_{0}与每一个h_{i}代入s(a,b)计算匹配度, 并做softmax处理拿到概率分布A$

所以$C_{0}=\sum_\limits{i}h_{i}A_{i}=h^{T}\cdot A$

于是$Y_{0}=f(C_{0},X_{1})$

seq2seq中使用RNN的问题: hard to parallel, 于是有人想用CNN替代, 当然也有人提出self-attention

![image-20210315184734598](Transformer/image-20210315184734598.png)

### Self-Attention Layer

![image-20210315184702743](Transformer/image-20210315184702743.png)
$$
q^{i}=W^{q} a^{i}\\
k^{i}=W^{k} a^{i}\\
v^{i}=W^{v} a^{i}
$$
<img src="Transformer/image-20210315190516684.png" alt="image-20210315190516684" style="zoom:50%;" />

### Multi-head Self-attention

<img src="Transformer/image-20210315192716075.png" alt="image-20210315192716075" style="zoom:50%;" />

<img src="Transformer/image-20210315192730176.png" alt="image-20210315192730176" style="zoom:50%;" />

多个head的原因是: 每个head的关注点不一样

<img src="Transformer/image-20210315195543763.png" alt="image-20210315195543763" style="zoom:50%;" />

### Bert: unsupervise trained transformer, Encoder of Transformer

bert有24层, 48层

| 训练方法1: Masked LM<br/><br/>15 % 词汇被随机置换为[Mask], 要求bert把这15%词汇填回来, 即把[MASK]通过bert运行后返回的对应embedding交给linear multi-class classifier用于预测被masked的是谁.<br/><br/>如果两个词天在同一个地方没有违和感, 那他们就有相似的emebdding. e.g. "潮水退了就知道谁没穿裤子", "潮水落了就知道谁没穿裤子", "退了"与"落了"就会有相似的embedding | 训练方法2: Next Sentence Prediction(给两个句子判断他们是连在一起的or not)<br/><br/>[CLS]醒醒吧[SEP]你没有妹妹, [SEP]: the boundary of two sentences [CLS]告诉bert需要做分类的事情<br/><br/>把[CLS]经过bert返回的embedding放入linear binary classifier, 这个clkassifier返回yes/no判断接下来的两个句子 |
| ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| <img src="Transformer/image-20210315202257845.png" alt="image-20210315202257845" style="zoom:50%;" />                                                                                                                                                       | <img src="Transformer/image-20210315202243447.png" alt="image-20210315202243447" style="zoom:100%;" />                                                                                                                                    |

不考虑一开始的位置向量的话, 对于self-attention来说, 一个词/句子无论是放在那里都没有影响, 都要评估它与所有其他词的关系

Linear binary classifier 与bert一同被训练

#### 实际操作

approach1与2同时被使用用来训练bert, 这样学的最好

问题来了, 怎么用bert呢?

### bert的应用

| 一般情况                                                                | case 2 预测句子中的词汇的词性                                                                                     | case 3 输入两个句子, output一个class, e.g. 给出"前提", 判断"假设"是否成立               | case QA system                                                                  |
| ------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------- | ------------------------------------------------------------------------------- |
| ![image-20210315203126480](Transformer/image-20210315203126480.png) | <img src="Transformer/image-20210315203542310.png" alt="image-20210315203542310" style="zoom:100%;" /> | ![image-20210315204034002](Transformer/image-20210315204034002.png) | ![image-20210315204251333](Transformer/image-20210315204251333.png)             |
|                                                                     |                                                                                                        |                                                                     | 学出两个vector, liangge vector与bert返回的emberrding做dot product, 接着softmax处理拿到p distri |

在这些应用中, bert用与训练模型, 但需要用fine-tune进行微调, classifier需要自己训练, 二者同时进行

### Bert参考文章 1 [理解BERT：一个突破性NLP框架的综合指南](https://www.jiqizhixin.com/articles/2019-11-05-2)

BERT架构建立在Transformer之上。我们目前有两个可用的版本:

- BERT Base:12层transformer，12个attention heads和1.1亿个参数
- BERT Large:24层transformer，16个attention heads和3.4亿个参数

输入:

综合的Embedding方案包含了很多对模型有用的信息

![img](Transformer/640.png)

1. **位置嵌入(Position Embeddings)**:BERT学习并使用位置嵌入来表达句子中单词的位置。这些是为了克服Transformer的限制而添加的，Transformer与RNN不同，它不能捕获“序列”或“顺序”信息
2. **段嵌入(Segment Embeddings)**:BERT还可以将句子对作为任务的输入(可用于问答)。这就是为什么它学习第一和第二句话的独特嵌入，以帮助模型区分它们。在上面的例子中，所有标记为EA的标记都属于句子A(对于EB也是一样)
3. **目标词嵌入(Token Embeddings)**:这些是从WordPiece词汇表中对特定词汇学习到的嵌入

# Domain Adaptation

训练集与测试机有较大差异时

| 原因                                                                  | 目的                                                                  |
| ------------------------------------------------------------------- | ------------------------------------------------------------------- |
| ![image-20210315232400403](Transformer/image-20210315232400403.png) | ![image-20210315232417647](Transformer/image-20210315232417647.png) |

### Discrepancy based

通过与先拿到的统计学上的数据, 来计算target&source上的距离. 希望他们的距离尽可能的小

| Deep Domain Confusion                                               | Deep Adaptation Networks                                            | CORAL: use 2nd order moments                                        | CMD: use even higher moments                                        |
| ------------------------------------------------------------------- | ------------------------------------------------------------------- | ------------------------------------------------------------------- | ------------------------------------------------------------------- |
| ![image-20210315232937626](Transformer/image-20210315232937626.png) | ![image-20210315233136428](Transformer/image-20210315233136428.png) | ![image-20210315233251025](Transformer/image-20210315233251025.png) | ![image-20210315233436812](Transformer/image-20210315233436812.png) |
| 最小化source domain上的classification loss与source&target之间的差异            | 与之前一样, 但是从1层变为了30层                                                  | 前面的算法本职位1st order moments计算距离, 这里用2nd order moments计算距离             |                                                                     |

### Adversarial based

| **Simultaneous Deep Transfer Across Domains and Tasks**                                               | **Domain Adversarial Training of Neural Networks**                                                     | PixelDA                                                                                               |
| ----------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------- |
| <img src="Transformer/image-20210316012835429.png" alt="image-20210316012835429" style="zoom:50%;" /> | <img src="Transformer/image-20210316013639871.png" alt="image-20210316013639871" style="zoom:100%;" /> | <img src="Transformer/image-20210316014443853.png" alt="image-20210316014443853" style="zoom:50%;" /> |
| ![image-20210316012857478](Transformer/image-20210316012857478.png)                                   |                                                                                                        |                                                                                                       |
| [参考](https://zhuanlan.zhihu.com/p/30621691)                                                           | 绿色:extarctor,特征提取<br/>蓝色:classifier,属于什么label<br/>红色:binary classification,来源于source/target            |                                                                                                       |

### Reconstruction based

这讲的不太好 -_-

## Application

1. Image to Image Translation
   
   Cross-Domain Image Translation
   
   * Unsupervised Image-to-Image Translation Networks (UNIT)
   
   * | -                                                                   | -                                                                   | -   |
     | ------------------------------------------------------------------- | ------------------------------------------------------------------- | --- |
     | ![image-20210316024815651](Transformer/image-20210316024815651.png) | ![image-20210316024825449](Transformer/image-20210316024825449.png) |     |
     
     |                                                                     | -                                                                   | -                                                                   |
     | ------------------------------------------------------------------- | ------------------------------------------------------------------- | ------------------------------------------------------------------- |
     | ![image-20210316025038341](Transformer/image-20210316025038341.png) | ![image-20210316025046653](Transformer/image-20210316025046653.png) | ![image-20210316025058735](Transformer/image-20210316025058735.png) |
   
   * $\mathcal{L}_{\mathrm{VAE}_{1}}\left(E_{1}, G_{1}\right)=\lambda_{1} \operatorname{KL}\left(q_{1}\left(z_{1} \mid x_{1}\right) \| p_{\eta}(z)\right)-\lambda_{2} \mathbb{E}_{z_{1} \sim q_{1}\left(z_{1} \mid x_{1}\right)}\left[\log p_{G_{1}}\left(x_{1} \mid z_{1}\right)\right]$
     
     $\mathcal{L}_{\mathrm{GAN}_{1}}\left(E_{2}, G_{1}, D_{1}\right)=\lambda_{0} \mathbb{E}_{x_{1} \sim P_{X_{1}}}\left[\log D_{1}\left(x_{1}\right)\right]+\lambda_{0} \mathbb{E}_{z_{2} \sim q_{2}\left(z_{2} \mid x_{2}\right)}\left[\log \left(1-D_{1}\left(G_{1}\left(z_{2}\right)\right)\right)\right]$
     
     $\left.\mathcal{L}_{\mathrm{CC}_{1}}\left(E_{1}, G_{1}, E_{2}, G_{2}\right)=\lambda_{3} \mathrm{KL}\left(q_{1}\left(z_{1} \mid x_{1}\right) \| p_{\eta}(z))+\lambda_{3} \mathrm{KL}\left(q_{2}\left(z_{2} \mid x_{1}^{1 \rightarrow 2}\right)\right)\right| \mid p_{\eta}(z)\right)-\\\lambda_{4} \mathbb{E}_{z_{2} \sim q_{2}\left(z_{2} \mid x_{1}^{1 \rightarrow 2}\right)}\left[\log p_{G_{1}}\left(x_{1} \mid z_{2}\right)\right]$
   
   MUNIT

2. Semantic Segmentation
   
   AdaptSegNet
   
   ![image-20210316025232281](Transformer/image-20210316025232281.png)
   
   ![image-20210316025242869](Transformer/image-20210316025242869.png)
   
   CBST 详见pdf

3. Person Re-ID
   
   SPGAN
   
   ECN

[^padding]: Padding As we know all the neural networks needs to have the inputs that should be in similar shape and size. When we pre-process the texts and use the texts as an inputs for our Model. Note that not all the sequences have the same length, as we can say naturally some of the sequences are long in lengths and some are short. Where we know that we need to have the inputs with the same size, now here padding comes into picture. The inputs should be in same size at that time padding is necessary. [cite from](https://www.dezyre.com/recipes/what-is-padding-nlp#:~:text=Recipe%20Objective-,What%20is%20padding%20in%20NLP%3F,an%20inputs%20for%20our%20Model.)
