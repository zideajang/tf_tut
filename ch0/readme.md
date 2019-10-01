
- 第一天 基础（前夜）
- 第二天 CNN
- 第三天 CNN
- 第四天 优化
- 第五天 RNN
- 第六天 实例
- 第七天 实例



今天是国庆节，首先祝福一下我们伟大的祖国，也希望自己以后在这么好的大环境下有好的发展。
![祝福祖国](https://upload-images.jianshu.io/upload_images/8207483-d8fe56e9da2f7aeb.jpg?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
在世界范围的机器学习的热度图来看，中国对机器学习热度完全不逊于美国，可能还略胜一筹。有这么好大环境，而且人工智能是大势所趋，所以静下来心来学习学习机器学习。

![前夜](https://upload-images.jianshu.io/upload_images/8207483-9906c7324710cf5e.jpg?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

### 学习方法
我们面对一些比较难，当然并不是难离谱的知识，学起来很容就会放弃。我想机器学习对多数数学基础相对薄弱的人，很容易就会放弃。因为机器学习既有深度又有广度。不过这里给出一点个人学习经验，首先需要有兴趣，有了兴趣才能够坚持，然后我们要多看，第一遍看不懂没啥再看一遍再听一遍，看的多了自然也就是感觉，有了感觉我们自己深入弄懂这些知识点，最后就是要多鼓励自己。

![目标](https://upload-images.jianshu.io/upload_images/8207483-fbac69fbcd13f5d4.jpg?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)


### 目标
通过一起学习，从小白成长为可以运用神经网络自己做一些项目。深度神经网发展今天，有许多框架（例如 TensorFlow mxnet 和 keras ）供我们选择使用，机器学习也逐步得到应用和普及。真真切切走到我们生活中，我们也能亲身体会到深度学习给我们带来的便利。

### 准备
需要了解一些线性代数和高等数学知识，学过就行这样应该捡起来不难，不用系统学习，现用现学吧。不会涉及到特别高深难懂知识，所以有点心理准备。

### 主流框架
有了这些框架我们就可以轻松地搭建出模型，也就是设计出神经网络。我们就像玩乐高一样随意选择和组合搭建出神经网络，通过来调试各种参数，调整隐藏层顺序和组合来得到较高精准率，当然学习好基础知识和了解其实现的背后原理是必要，这样我们才可以轻松调整参数来得的预期的模型。
- TensorFlow
- MXNet
- Keras
在分享中实例主要是以上 3 三种来演示，不过大家知道除此之外当下流行框架还很多，例如Caffe、Torch 等。

## 基本术语
### 误差
我们通过训练的模型，然后通过对比模型得到预期值和实际值之间差距大小来评估模型好坏，那么为什么会有误差呢？误差是从哪里来，只有搞懂这些才能够帮助我们调整参数来得到好的模型。
假设期望值（也就是实际值）我们用 $\hat{y}$ 表示真实函数（也就是数据背后规律），我们需要通过训练来的得到一个比较接近$\hat{y}$ ，我们用 $y^*$ 来表示。

概率与统计
我们知道 $\mu$ 是所有数据的平均值，而我们计算其中样本数据，也就是全体数据一部分的数据是
$$ m = \frac{1}{N} \sum_nx^n $$  
$$  \mu $$

$$ E[m] = E[\frac{1}{N}\sum_n x^n] = \frac{1}{N}\sum_n E[x^n] = \mu $$

$$ Var[m] = \frac{\sigma^2}{N}$$
我们数据是否分散是和我们训练数据集大小有关的。这个很好理解数据量越大我们样本的并均值 m 就越接近真实均值 $\mu$

$$ m = \frac{1}{N} \sum_nx^n $$ 
$$ s^2 = \frac{1}{N} \sum_n (x^n - m)^2 $$ 
$s^2$ 是估计值会散步 $\sigma^2$ 周围
$$ E[s^2] = \frac{N-1}{N} \sigma^2  $$

通过图我们不难看出误差来源两件事，第一个就是我们估测值点中心值是否偏离靶心(Bias)，而第二个是我们这些估计值分散程度(Variance)，也就是训练稳定性。

### 损失函数（代价函数)


### 梯度下降
最小化 
初始化 $L(\theta_0,\theta_1)$
不断更新 $\theta_0, \theta_1$

- 红色区域表示损失函数比较大
- 蓝色区域表示损失函数比较小
不断迭代过程就是求导，求导后就会找到一个方向，朝着梯度下降方向移动来更新$\theta_1 \theta_0$

问题根据选取初始值可能会走到局部极小值而不是全局最小值。

$$ w^1 \leftarrow w^0 - \eta\frac{dL}{dw} | _{w=w^0}$$
$$ w^2 \leftarrow w^1 - \eta\frac{dL}{dw} | _{w=w^1}$$
- $\eta$ 表示学习率
- $w^0$ 表示初始值参数
- $\eta\frac{dL}{dw}$ 表示损失函数 L对于参数 w 取导数

当切线为负，我们就增加 w 值像右移动 $\eta$ 是学习速度，也就是移动步长。知道走到微分为 0 时候我们停止前进，所以我们就会思考梯度下降不一定会知道最小值。

上面考虑是一个参数情况，接下来我们来考虑 2 参数
$$ \frac{\partial L}{\partial w} | _{w=w^0,b=b^0},\frac{\partial L}{\partial b} | _{w=w^0,b=b^0} $$
![两个参数梯度下降](https://upload-images.jianshu.io/upload_images/8207483-0145f3d48e89c8ec.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

$$ w^1 \leftarrow w^0 - \eta\frac{dL}{dw} | _{w=w^0,b=b^0}$$
$$ w^2 \leftarrow w^0 - \eta\frac{dL}{dw} | _{w=w^0,b=b^0}$$

$$ b^1 \leftarrow b^0 - \eta\frac{dL}{dw} | _{w=w^1,b=b^1}$$
$$ b^2 \leftarrow b^1 - \eta\frac{dL}{dw} | _{w=w^1,b=b^1}$$

$$
\Delta
$$
这个线就是等高线的法线方向，然后更新参数，每一次根据偏微分来移动点来获取下一次参数 w 和 b

$$ \theta^* = \arg \max_{\theta} L(\theta)$$
 
 $$ L(\theta^0) > L(\theta^1) > L(\theta^2) > \cdots$$

 ###
 $$ f(x) = wx + b$$
 $$ f(x) = \frac{1}{2m}\sum_{i=1}^m(f(x^i) - y^i)^2 $$


 $$ \frac{\partial L(w,b)}{\partial w} = \frac{1}{m} \sum_{i=1}^m(f(x^i) - y^i) x^i $$

$$ \frac{\partial L(w,b)}{\partial b} = \frac{1}{m} \sum_{i=1}^m(f(x^i) - y^i)$$
### 总结
我们关注可以调整参数这里就是学习率 $\eta$
- 梯度下降是同步更新
- 学习率不能太小也能太大
- 学习率太大，每一步太长就会出现在函数谷底附近进行震荡而无法达到最小值
- 代价函数式突函数，非凸函数有许多局部极小值



### 反向传播
![神经元](https://upload-images.jianshu.io/upload_images/8207483-15a78831c6349e98.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

神经元组成
- 细胞核
- 树突
- 细胞体
- 轴突
- 神经末梢
树突接收其他神经元发送信息，处理后通过神经末梢将信息传递给其他神经元。神经元网络就是受到生物神经网络的启迪而来。接下来我们将这个神经元抽闲出来深度网络的最基本单元。
![神经网络基本计算结构](https://upload-images.jianshu.io/upload_images/8207483-2e5aed32480d5b21.jpg?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

这么简单的结构可能用处不大，但是大数力量让我们看到神经网络的威力，这是这样上亿上千万的简单结构组合在一起让神经网络今天大放异彩。
![大放异彩](https://upload-images.jianshu.io/upload_images/8207483-7ccc1d98a38ae3e0.jpg?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

![全连接网络层之间的基本计算方式](https://upload-images.jianshu.io/upload_images/8207483-b6e82f7f00fba519.jpeg?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
输入层(input layer)为数据输入，输出层(output layer)

$$ a_1 = w_1x_1 + w_2x_2 + w_3x_3 + b_1 $$

$$ a_1 = w_11x_1 + w_12x_2 + w_13x_3 + b_1 $$
$$ a_2 = w_1x_1 + w_2x_2 + w_23x_3 + b_1 $$
$$ a_3 = w_1x_1 + w_2x_2 + w_3x_3 + b_1 $$


### 反向转播
这是一个比较难于理解的部分，不过如果你不理解似乎也不会影响你使用机器学习，不过只有很好理解什么是反向传播才能够真正理解在模型中是如何优化参数的。

坐下来想一想，我们通过损失函数可以查看出估计值与期望值之间差距来评估当前参数好坏。知道了我们参数（也就是函数）好坏就是为了优化函数，我们优化参数具体工作就是更新参数。更新那些参数以及更新幅度是如何操作的呢。
我们需要通过损失函数对每一个参数的导数，简单有效方法就是反向求导。

#### 求导链式法则
我们先回忆一下什么是链式求导，这是在大学学过的知识。
- 第一种情况
$$ y = g(x) z = h(y) $$
$$ \Delta x \rightarrow \Delta y \rightarrow \Delta z$$ 

$$ \frac{dz}{dx} = \frac{dz}{dy} \frac{dy}{dx} $$

当 x 发生变化就会影响到 y ，那么当 y 发生变化就会影响 z，那么也就是 x 间接通过 y 来影响 z。


- 第二种情况

$$  x = f(s)  y = g(s) z = k(x,y) $$
$$ \Delta s \rightarrow \Delta x $$
$$ \Delta s \rightarrow \Delta y $$
$$ \Delta x \Delta y \rightarrow \Delta y $$

$$ \frac{dz}{dx} = \frac{\partial z}{\partial x} \frac{dx}{ds} + \frac{\partial z}{\partial y} \frac{dy}{dx}$$

这里我们看到当 s 发生变化就会影响到 x 和 y，而 x 和 y 发生变化就影响 z ，也可以这样看 s 通过 x 和 y 两条路径可以到达 z。

$$ X^n \rightarrow \underline{NN}  \rightarrow  y^n \rightarrow_{l^n} \hat{y}^n $$

![图](https://upload-images.jianshu.io/upload_images/8207483-a1488c4ce9bc3a1d.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)



$$ L(\theta) = \sum_{n=1}^N l^n(\theta) \rightarrow \frac{\partial L(\theta)}{\partial{w}} = \sum_{n=1}^N \frac{\partial l^n(\theta)}{\partial{w} } $$
我们对 $L(\theta)$ 求导也就是 $L(\theta)$ 是每一 $l^n(\theta)$

![正向传播](https://upload-images.jianshu.io/upload_images/8207483-54ed881e2543a493.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

$$ \frac{\partial z}{\partial w_1} = x_1 $$
$$ \frac{\partial z}{\partial w_2} = x_2 $$

在图中当我们对 z 对 $W_1$ 进行偏微分得到值就是 z 输入 $x_1$ 的值。

![正向传播](https://upload-images.jianshu.io/upload_images/8207483-cd3b17a60e7b7bf2.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

正向传播是分简单的


#### 反向传播算法的思路
- 先计算**向前传播**

$$
    = \frac{\delta fr}{1}
$$

### 优化
### 过拟合和正则化
#### 欠拟合(Underfitting)

$$ f(x) = wx + b  $$

也就是函数的复杂度不够无法
#### 过拟合(Overfitting)
$$ f(x) = w_1x^3 + w_2x^2 + w_3x + b $$
在训练集表示很好，因为参数过多所以函数精确拟合到训练集每一个数据，也就是函数记住了每一个数据，也就是对于简单问题我们选择复制模型。

#### 正则化
$$ f(x) = \frac{1}{2m} [ \sum_{i=1}^m(f(x^i) - y^i)^2 \lambda \sum_{i=1}^n w_i^2 ]$$

#### 总结
- 防止过拟合
    - 减少特征
    - 增加数据量，增加数据量也就是用更复杂的数据来匹配复杂模型
    - 正则化(Regularized)
    - dropout(随后深入讲解)


### 激励函数


## 回归问题
线性回归可以应用在股票预测，房价预测，无人车驾驶、还就就是推荐系统，淘宝或者京东，用户A购买商品 B 可能性也是。李宏毅老师是以预测宝可梦进化为例来讲的线性回归，所谓线性回归就是可以暂时理解根据以往经验（数据）总结出规律来预测未来。
再简化一些问题，其实

### 抛出问题
$$ f(x) = 3 \cdot x $$
今天我们要解决一个问题就是通过训练集数据来找到一个函数（上面函数），其实这个应该对于我们在简单不过了，学多人在小学时候就会通过带入数据来找到 x 权重 3。正式因为这是简单问题，我们自然就会解决而忽略解决问题过程。今天的神经网络也就是模拟我们人类是如何学习的。
当然我们也会遇到相对更复杂问题，
$$ f(x_1,x_2) = 2 x_1 + 3 x_2 + 5$$
今天我们是通过学习来找到这么方程然后，用方程来预测一些值，那么我们如何通过学习来找到这个**函数**呢。这也就是今天我们今天主要要学习的。

### 1.定义模型
什么是模型，所谓模型我们认为是一组函数的集合，所谓函数集也称函数族，并不是形状各异天马行空的函数，而是设计出具有一定样式，只是参数 w 和 b 不同而已，我们要做工作就是在函数集中找出最优函数（最好的 w 和 b）来拟合我们真实函数。所以第一步定义模型就是定义函数大概什么样式，几元几次函数呀。线性问题我们通常会定义为如下形式
$$ y = wx + b $$
也可以表示为这种形式
$$ f(x) = w \cdot x + b $$
在这个函数的参数 w 和 b 可能是任意数值，理论上根据 w 和 b 不同，函数数量是无穷。

$$
    f_1 = 2x + 2 
$$
$$
    f_2  = 8.8x + 1
$$
$$
    f_3  = 0.75x - 2
$$
不过为了简化问题我们也会将 w 和 b 限定在一定范围来减少运算。
这是一个线性模型，我们需要进行初步筛选，为什么说这是线性函数，这是因为函数可以写出这种形式
$$
    y = \sum x_iw_i
$$
所以就是线性函数。这里 w 叫做权重 b 偏移值。

$$ (x^1,\hat{y}^1) (x^2,\hat{y}^2) \cdots (x^i,\hat{y}^i)$$
我们需要使用上标表示一个类，用下标表示该类的某个属性。

### 评估

评估我们函数好坏，这里定义一个损失函数（用L表示)，损失函数是接受我们一个模型一个函数作为参数，那么学过函数编程都知道这就是所谓高阶函数，我们f(x) 是 L的参数。这个函数输出就是通过数值表示这个函数有多好，和多不好。

所以引入损失函数(Loss Function) 也有人使用 Cost Function ,损失函数也会根据具体情况而不同

$$ L(f(x))= L(b,w)$$
因为我们这里 f(x) 是 b 和 w 决定的，所以我们就可以用 L(b,w)

$$ L(f) = \sum_{i=1}^n(\hat{y}^n = (b + w \cdot x^n ))^2 $$

- $\hat{y}^n$ 是真实值，我们现在函数计算出值与真实值做差后平方，来计算每一个样本的差值，然后在取平方后求和。

其实我们要弄清两个问题
- 第一个问题是优化谁：我们需要不但更新权重 w 和偏差值 b 来找最优函数
- 第二个问题是怎么优化：通过缩小预期值和真实值的误差来优化。

### 优化

$$ f^* = arg \min_{w,b} L(w,b) $$
$$ w^*,b^* = 
arg \min_{w,b} \sum_{n=1}^10 ( \hat{y}^n - (b + w \cdot x_i^n) )^2 $$ 
这就是一个如何找到最优解的方程，我们知道求解这个方程就可以得到我们想要最优函数，也就是我们想要的 w 和 b。我们可以根据大学学过知识来解这个问题。

#### 梯度下降（大名鼎鼎）
梯度下降是用于处理一切可微分的方程，我们需要在曲线上找到最低点，也就是导数为 0 位置，不过导数为 0 位置不一定是最低点，随后案例我们就会遇到这种情况。
- 随机找到一位置
$$  \frac{dL}{dx} | _{w=w^0} $$

我们在曲线随意找一点作为起始点，然后在这点进行求导得到关于曲线这一点的切线。
![一个参数梯度下降](https://upload-images.jianshu.io/upload_images/8207483-91594f2419522cf6.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
$$ w^1 \leftarrow w^0 - \eta\frac{dL}{dw} | _{w=w^0}$$
$$ w^2 \leftarrow w^1 - \eta\frac{dL}{dw} | _{w=w^1}$$
当切线为负，我们就增加 w 值像右移动 $\eta$ 是学习速度，也就是移动步长。知道走到微分为 0 时候我们停止前进，所以我们就会思考梯度下降不一定会知道最小值。

上面考虑是一个参数情况，接下来我们来考虑 2 参数
$$ \frac{\partial L}{\partial w} | _{w=w^0,b=b^0},\frac{\partial L}{\partial b} | _{w=w^0,b=b^0} $$
![两个参数梯度下降](https://upload-images.jianshu.io/upload_images/8207483-0145f3d48e89c8ec.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

$$ w^1 \leftarrow w^0 - \eta\frac{dL}{dw} | _{w=w^0,b=b^0}$$
$$ w^2 \leftarrow w^0 - \eta\frac{dL}{dw} | _{w=w^0,b=b^0}$$

$$ b^1 \leftarrow b^0 - \eta\frac{dL}{dw} | _{w=w^1,b=b^1}$$
$$ b^2 \leftarrow b^1 - \eta\frac{dL}{dw} | _{w=w^1,b=b^1}$$

$$
\Delta
$$
这个线就是等高线的法线方向，然后更新参数，每一次根据偏微分来移动点来获取下一次参数 w 和 b

$$ \theta^* = \arg \max_{\theta} L(\theta)$$
 
 $$ L(\theta^0) > L(\theta^1) > L(\theta^2) > \cdotss$$

 #### 问题点
 ![梯度下降停止点](https://upload-images.jianshu.io/upload_images/8207483-1b39abe32aa653d5.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

$$
    L(w,b) = \sum_{n=1}^{10}( \hat{y}^n - (b + w \cdot x_i^n))^2

$$

$$
\frac{\delta L}{\delta w} = \sum_{n=1}^{10}2( \hat{y}^n - (b + w \cdot x_i^n))(-x_i^i)
$$


$$
\frac{\delta L}{\delta b} = \sum_{n=1}^{10}2( \hat{y}^n - (b + w \cdot x_i^n))(-1)
$$

### 线性回归实例
来一个线性回归问题实例，在开始之前我们先明确一些问题，也就是我们要训练模型几个步骤，在李宏毅教授的课程中他提到会分为建立模型（也就是函数集合），然后就是定义损失函数，最后根据损失函数进行优化找到最优的函数。
不过我们对于简单的神经网，个人总结了一点经验，首先是准备好训练和测试数据集，然后处理这些数据以便训练，然后就是定义输入变量，因为都是矩阵和向量乘法，在设计神经网网络时候我们需要考虑结构以及矩阵的维度和矩阵元素类型，这样运算可以顺利进行。然后就是定义损失函数，接下来是定义优化找到让损失函数最小的那个方程也就是我们最优解，然后用最优解方程来测试我们测试数据，来评估我们模型精确度。
#### 准备数据
```python
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

learning_rate = 0.01
epochs = 200

n_samples = 30
train_x = np.linspace(0,20,n_samples)
train_y = 3 * train_x + 4* np.random.rand(n_samples)

plt.plot(train_x,train_y,'o')
plt.show()
```

```python
plt.plot(train_x,train_y,'o')
plt.plot(train_x, 3 * train_x)
plt.show()
```

#### 创建变量

```python
X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

W = tf.Variable(np.random.randn,name='weights')
B = tf.Variable(np.random.randn,name='bias')
```

#### 定义计算图
$$ pred = X * W + B $$

#### 定义损失函数
然后定义评估函数也就是我们提到损失函数
$$ \frac{1}{2n} \sum_{i}^n(pred(i) -Y_i)^2  $$

```python
cost = tf.reduce_sum((pred - Y) ** 2) / (2 * n_samples)

```
用计算出估计值(pred) 减去实际值，也就是估计值到真实值的距离然后平方来去掉负数后取均值。

#### 优化
```
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
```
我们优化就是需要让损失函数最小，所以最小化 cost 函数的参数就是最好的参数
#### 训练

```python
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    for epoch in range(epochs):
        for x, y in zip(train_x,train_y):
            sess.run(optimizer,feed_dict={X:x,Y:y})
        if not epoch % 20:
            c = sess.run(cost,feed_dict={X:train_x,Y:train_y})
            w = sess.run(W)
            b = sess.run(B)
            print(f'epoch:{epoch:04d} c={c:.4f} w={w:.4f} b={b:.4f}')
```
设计好我们计算图后，就可以开始进行计算了，我们所有计算图都需要放到 

#### 验证结果
```
    weight = sess.run(W)
    bias = sess.run(B)
    plt.plot(train_x,train_y,'o')
    plt.plot(train_x,weight * train_x + bias)
    plt.show()
```

```
epoch:0000 c=78.2851 w=1.8528 b=1.4544
epoch:0020 c=7.5476 w=2.8999 b=1.4118
epoch:0040 c=7.4682 w=2.9079 b=1.2879
epoch:0060 c=7.3967 w=2.9155 b=1.1703
epoch:0080 c=7.3324 w=2.9226 b=1.0587
epoch:0100 c=7.2745 w=2.9295 b=0.9528
epoch:0120 c=7.2224 w=2.9359 b=0.8522
epoch:0140 c=7.1755 w=2.9421 b=0.7568
epoch:0160 c=7.1334 w=2.9479 b=0.6662
epoch:0180 c=7.0954 w=2.9535 b=0.5802
```

### 完整神经网络