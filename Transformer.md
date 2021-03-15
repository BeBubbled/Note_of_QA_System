[toc]

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

| -                                                            | -                                                            |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| ![image-20210314220632382](Transformer/image-20210314220632382.png) | ![image-20210314220641471](Transformer/image-20210314220641471.png) |

假设现在i==1

所以我们手里有$decoder隐藏层H_{0}, 权重计算函数s(a,b),encoder的每一个隐藏层h_{i}$

所以我们对$H_{0}与每一个h_{i}代入s(a,b)计算匹配度, 并做softmax处理拿到概率分布A$

所以$C_{0}=\sum_\limits{i}h_{i}A_{i}=h^{T}\cdot A$

于是$Y_{0}=f(C_{0},X_{1})$



seq2seq中使用RNN的问题: hard to parallel, 于是有人想盗用CNN替代





Bert: unsupervise trained transformer



# Deep learning for Question Answering System



# Doamin Adaptation





[^padding]: Padding As we know all the neural networks needs to have the inputs that should be in similar shape and size. When we pre-process the texts and use the texts as an inputs for our Model. Note that not all the sequences have the same length, as we can say naturally some of the sequences are long in lengths and some are short. Where we know that we need to have the inputs with the same size, now here padding comes into picture. The inputs should be in same size at that time padding is necessary. [cite from](https://www.dezyre.com/recipes/what-is-padding-nlp#:~:text=Recipe%20Objective-,What%20is%20padding%20in%20NLP%3F,an%20inputs%20for%20our%20Model.)

