## Gan

generator+ranfdom vector-> target high dimensional vector

discriminator:像二次元则高分, 否则低分



不仅训练gnererator还需要discriminator





### Conditional Gan (Supervised)

generator有可能会发现某一种一旦可以骗过discriminator后, 就不再改变自己, 无视输入, 时钟输出同一个东西来欺骗discirminator

现在我们修改discirminator, 不仅判断generator的结果有多好, 还判定generator的输入与输出有多匹配

1. text->image



​	好图片+好text=1

​	好图片+烂text=0=兰图片+好text

2. sound to image

​	e.g. 电视雪花声->瀑布,声音越大, 瀑布越猛

​	类似直升机的声音->快艇海上行走, 声音越大, 快艇引起的水花越大

3. image->text

   e.g. image-> multi label

## Unsupervised Gan Cycle Gan



## Hung-yi Lee Generative Adversaria

Generator: a neural network

![](https://raw.githubusercontent.com/BeBubbled/PicGoImages-WorkSpace/master/image-20210312100521080.png?token=ANU6SUC6PFFLAXAV6AID76DAJWRJC)

![](https://raw.githubusercontent.com/BeBubbled/PicGoImages-WorkSpace/master/image-20210312100538459.png?token=ANU6SUD7K25OCO2NPVNERODAJWRGM)

## Gan discriminator predict

假设数据集内部元素呈现线性分布, 于是可以遍历的方式拿到所有可能的数据集, 这些数据及在discriminator中分数最高的即为预测值, 这些数据集中, 属于training的应该让dircriminator给出高分, 不属于training的dircriminator应该给出低分, 借此完成discirminatro的独立training



## Gan Feature Extraction

### infoGan

![image-20210314003926246](March_14/image-20210314003926246.png)

![image-20210314003935532](March_14/image-20210314003935532.png)

假设我们打算生成像MNIST那样的手写数字图像，每个手写数字可以分解成多个维度特征：代表的数字、倾斜度、粗细度等等，在标准GAN的框架下，我们无法在上述维度上具体指定Generator生成什么样的手写数字。

为了解决这一问题，文章对GAN的目标函数进行了一些小小的改进，成功让网络学习到了可解释的特征表示（即论文题目中的interpretable representation）。

[infoGan理解](https://zhuanlan.zhihu.com/p/58261928)

### VAE-GAN

![image-20210312134055741](March_14/image-20210312134055741.png)

![image-20210312134417230](March_14/image-20210312134417230.png)

## BiGan

![image-20210312133943456](March_14/image-20210312133943456.png)

![image-20210312134358537](March_14/image-20210312134358537.png)

![image-20210312140523483](March_14/image-20210312140523483.png)

让encoder与decode越相似越好

Bigan得到的auto encoder与一半的auto-encoder特性不一样



### Triple Gan

### Domain-adversarial training

![image-20210314000857786](March_14/image-20210314000857786.png)

