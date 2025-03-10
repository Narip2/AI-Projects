Mainly refer to [Nielsen's book](https://static.latexstudio.net/article/2018/0912/neuralnetworksanddeeplearning.pdf).

# Gradient descent
- Batch Gradient Descent
- Stochastic Gradient Descent
- Mini-Batch GD

Consider if we have $n$ inputs in total. 

Two references:
[Video 1](https://www.bilibili.com/video/BV13p4y1g7eQ/?vd_source=ac9b07978062a2dbae3c01bd0e801738)
[Blogs](http://cnblogs.com/lliuye/p/9451903.html)

# Backpropagation

反向传播的核心其实就是链式求导法则。不过为了编程易于实现，并且复用链式求导法则中的中间导数，我们需要对链式求导法则进行稍微的修改。为了实现前面所说的目的，我们需要定义几个符号： 

$$
w_{ij}^{\left( l \right)},b_{i}^{\left( l \right)},a_{i}^{\left( l \right)},z_{i}^{\left( l \right)}.
$$

这里，$w_{ij}^{\left( l \right)}$ 表示的是从 $l-1$层到 $l$ 层的 weights。其中下标 $ij$ 表示从 $l-1$层的第 $j$个神经元映射到第 $l$ 层的第 $i$个神经元。这里的 $b_{i}^{\left( l \right)}$ 表示的是第 $l$ 层的第 $i$ 个 bias。 $a_{i}^{\left( l \right)}$ 表示的是第 $l$层第 $i$ 个神经元上的 activation, 而 $z_{i}^{(l)}$ 表示的是第 $l$ 层第 $i$ 个神经元上的 $z$函数， $z$函数满足关系 $a_{i}^{\left( l \right)}=\sigma \left( z_{i}^{\left( l \right)} \right)$。Here, $\sigma$ stands for the sigmoid function.

All the formulas will be used in the backpropagation procedure can be found in the following picture, where $\delta$ is the newly defined quantity that will simplify our computation.

![[Pasted image 20250309220500.png]]

# CNN实现
CNN(一个简短的[介绍视频](https://www.bilibili.com/video/BV1MsrmY4Edi?spm_id_from=333.788.videopod.episodes&vd_source=ac9b07978062a2dbae3c01bd0e801738)) 的基本思想包含以下三个：
- local receptive fields
- shared weights (also called kenerl or filter)
- pooling

第一步：
![[Pasted image 20250306215938.png]]
网络里面的输出是用下面这个映射：

$$
\sigma\left(b+\sum_{l=0}^4 \sum_{m=0}^4 w_{l, m} a_{j+l, k+m}\right)
$$
这里的 $w$ 和 $b$ 被称作shared weights和 shared biase. 如果有多个kernel，就会得到多个卷积的处理结果，如下图的三个kernel所示：
![[Pasted image 20250307081915.png]]
第二部就是池化(pooling)。池化的一个显著好处就是数据被压缩。常用的池化操作有max-pooling和L2 pooling。
![[Pasted image 20250307082113.png]]

# Use MNIST dataset

