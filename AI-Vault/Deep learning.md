Mainly refer to [Nielsen's book](https://static.latexstudio.net/article/2018/0912/neuralnetworksanddeeplearning.pdf).

# Gradient descent
- Batch Gradient Descent
- Stochastic Gradient Descent
- Mini-Batch GD

Two references:
[Video 1](https://www.bilibili.com/video/BV13p4y1g7eQ/?vd_source=ac9b07978062a2dbae3c01bd0e801738)
[Blogs](http://cnblogs.com/lliuye/p/9451903.html)

# Backpropagation

反向传播的核心其实就是链式求导法则。不过为了编程易于实现，并且复用链式求导法则中的中间导数，我们需要对链式求导法则进行稍微的修改。为了实现前面所说的目的，我们需要定义几个符号： 

$$
w_{ij}^{\left( l \right)},b_{i}^{\left( l \right)},a_{i}^{\left( l \right)},z_{i}^{\left( l \right)}.
$$
