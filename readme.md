### 章节名称
#### 节名称
##### 小节名称

### 特别感谢
想了想自己要是要感谢一些人，正式他们浅显感谢李宏毅教授的课程，部分图片和资料参考李宏毅教授的课程,也参照许多相关书籍。
感谢 sentdex 的分享。部分参照了一些书籍内容。

### 目标
通过一起学习，从小白成长为可以运用神经网络自己做一些项目。终极目标是通过学习，从而得到高薪职位改善一下自己生活。深度神经网发展今天，有许多框架（例如 TensorFlow mxnet 和 keras ）供我们选择使用，机器学习也逐步得到应用和普及。真真切切走到我们生活中，我们也能亲身体会到深度学习给我们带来的便利。



### 应用分析
- 物体识别
- 影像/语音识别
- 聊天机器人
- 其实翻译
#### 语言
百度地图搜索
$ \underline{ 沈阳师范大学 } 附近的楼盘 $
$$ \underline{ab} $$
- tokenize 分词
- embedding 就是将词转换为向量
- RNN （多层）
- 全连接（2层）
百度语音
- 小爱
- 小bing
难度是多轮对话
RNN
$$ h_t = \sigma(w \cdot h_{t_1} + u \cdot x_t $$

### 准备
需要了解一些线性代数和高等数学知识，学过就行这样应该捡起来不难，不用系统学习，现用现学吧。不会涉及到特别高深难懂知识，所以有点心理准备。
### 主流框架
有了这些框架我们就可以轻松地搭建出模型，也就是设计出神经网络。我们就像玩乐高一样随意选择和组合搭建出神经网络，通过来调试各种参数，调整隐藏层顺序和组合来得到较高精准率，当然学习好基础知识和了解其实现的背后原理是必要，这样我们才可以轻松调整参数来得的预期的模型。
- TensorFlow
- MXNet
- Keras
在分享中实例主要是以上 3 三种来演示，不过大家知道除此之外当下流行框架还很多，例如Caffe、Torch 等。

### 基础术语
- cpu
离散数据
- gpu
矩阵的计算适合

### 简单神经网络

$$ h_1 = \sigma(w_1 \cdot x + b_1) $$
$$ y = softmax(w_2 \cdot h + b_2)$$
$$ Loss(w) \sum y \prime logf(x)$$

### 监督学习
$$ (x_1,y_1) \cdots $$
$$ L(w) = \sum_{i=1}^N l(f(x_i,w),y_i) $$

### 搜索




沈阳到北京怎么走

### 生物神经网络

### 激活函数
$$ tanh(x) = \frac{e^{2x} - 1}{e^{2x} + 1}$$
通过图形我们会发现根据输入值很快输出 1 或者 -1 ，靠近 0 部分非常接近线性而在值比较大时候会有上下线控制。
$$ relu(x) = max(0,x) $$
这是是 tanh 简化版，计算器会比较快，在图像识别中推荐使用 relu 
$$ softmax(x)_i = \frac{e^{x_i}}{\sum e^{x_i}} $$
softmax 主要用来做一些分类问题，所有分类问题在最后一层输出都可以用 softmax

### 收敛
设计参数 learning-rate 过大也会导致模型不收敛

### 反向转播
这是一个比较难于理解的部分，不过如果你不理解似乎也不会影响你使用机器学习，不过只有很好理解什么是反向传播才能够真正理解在模型中是如何优化参数的。

坐下来想一想，我们通过损失函数可以查看出估计值与期望值之间差距来评估当前参数好坏。知道了我们参数（也就是函数）好坏就是为了优化函数，我们优化参数具体工作就是更新参数。更新那些参数以及更新幅度是如何操作的呢。
我们需要通过损失函数对每一个参数的导数，简单有效方法就是反向求导。



#### 求导链式法则
- 第一种情况
$$ y = g(x) z = f(h) $$
$$ \Delta x \rightarrow \Delta y \rightarrow \Delta z$$ 

$$ \frac{dz}{dx} = \frac{dz}{dy} \frac{dy}{dx} $$
参考一个老外课程来理解什么是**反向传播**，每一个函数可以看做管道阀来控制流量。

我们先回忆一下什么是链式求导。
- 第二种情况

$$  x = f(s)  y = g(s) z = k(x,y) $$
$$ \Delta s \rightarrow \Delta x $$
$$ \Delta s \rightarrow \Delta y $$
$$ \Delta x \Delta y \rightarrow \Delta y $$
#### 反向传播算法的思路
- 先计算**向前传播**

$$
    = \frac{\delta fr}{1}
$$


#### 实例
回顾一下求导的过程，$ y = sin (x^2 + 1 ) $
$$ \frac{\partial y}{\partial x } = \frac{\sin(x^2 +1)}{\partial x} = frac{\partial \sin(x^2 + 1)}{\partial (x^2 + 1)}$$
回到我们神经网络，

$$ X^n \rightarrow \underline{NN}  \rightarrrow  y^n \rightarrow \hat{y} $$


$$ L(\theta) = \sum_{n=1}^N l^n(\theta) \rightarrow \frac{\partial L(\theta)}{\partial{w}} = \sum_{n=1}^N \frac{\partial l^n(\theta)}{\partial{w} } $$

### 梯度下降
$$ \theta^* = arg \max_{\theta} L(\theta) $$
- L 损失函数
- $\theta$ 参数
我们需要找到一组参数 
$$ \lbrace \theta_1 \theta_2 \rbrace $$
首先我们随机选取一组参数
$$ \theta^0 = \left[
 \begin{matrix}
   \theta_1^0  \\
   \theta_2^0 
  \end{matrix}
  \right]  $$
我们将 $\theta^0_1$ 和 $\theta^0_1$ 表示
  $$ \left[
 \begin{matrix}
   \theta_1^1  \\
   \theta_2^1 
  \end{matrix}
  \right] = \left[
 \begin{matrix}
   \theta_1^0  \\
   \theta_2^0 
  \end{matrix}
  \right] - \eta \left[
 \begin{matrix}
   \frac{\partial L(\theta_1^0)}{\partial \theta_1}   \\
   \frac{\partial L(\theta_2^0)}{\partial \theta_2}  
  \end{matrix}
  \right]  $$

其中 $\theta^1$ 表示在 $\theta^0$ 更新后的到一组参数，更新方法如上公式。

提示
我们需要小心调整学习率(learning rate),随后我们用较小的学习率可能步伐太小，训练速度会放慢，放慢到我们无法忍受。如果 learning rate 过大我们就很容易错过损失函数最低点，而在最低点附近徘徊而始终无法到达损失函数的谷底。
那么我们怎么调整 learning rate，首先要明确 learning rate 不能是一层不变的。而且会时间（训练过程）而且不同参数也考虑给不同 learning rate 好总结一下就是 learning rate 设置需要考虑到时间和参数不同而不同。

$$ W^{t+1} \leftarrow W^t - \frac{\eta^t}{\theta^t} g^t$$
这里 $g^t$
$$ g^t = \frac{\partial L(\theta^t)}{\partial w} $$
也就是 $g^t$ 过去所有微分值的均方差。

#### Adagrad
$$ w^1 \leftarrow w^0 - \frac{\eta^0}{\sigma^0} g^0$$
$$  \sigma^0 = \sqrt{(g^0)^2}$$
过去所有计算过
$$ w^2 \leftarrow w^1 - \frac{\eta^1}{\sigma^1} g^1$$
$$  \sigma^1 = \sqrt{\frac{1}{2}[(g^0)^2 + (g^1)^2]}$$
从上面时间演变公式我们就不难看出 $g^t$ 是什么了吧，其实实际搭建神经网络这些工作不需要我们自己做，框架已经做好了提供方法调用就可以了，不过还是有必要来了解这些方法背后的原理。

### 分享特点
可能大家学习了学多机器学习课程，想要问你的分享有啥特点，就像我们在国美买手机，总会掏出手机查一下京东对应价格。我的特点就是我真真切切地从一个机器学习门外汉和我一起跨进到机器学习行业内。内容并不会避讳一些难点，虽然有些数学知识，也会给出详尽解释，如果你是一个专业人士估计对比不会有用。内容会不断随着自己学习更新和完善

### 前言
特别感谢李宏毅教程，可能内容一些资料和图来自于李宏毅老师的机器学习。也喜欢大家有时间多看一看李宏毅老师的课程，每一观看都会有新的收获。
#### 学习机器学习
#### 课程设计
整个内容都是伴随自己学习过程沉底下来，学习前自己状态是高数自从毕业后除了像考验看了一阵大概有十多年没有碰高数，
#### 为什么学习机器学习
- 机器学习有一定难度
- 机器学习是需求
- 
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
什么是模型，所谓模型我们认为是一组函数的集合，
$$ y = wx + b $$
这里我们定义函数模型，因为是一组函数组成模型，当然这些函数未知数和次元是我们设计的，所以我们定义模型函数如下，通过不同 w 和 b 来得到一组函数。
$$ f(x) = w \cdot x + b $$
在这个函数的参数 w 和 b 可能是任意数值，随意理论上我们函数数量是无穷。不过为了简化问题我们也会将 w 和 b 限定在一定范围来减少运算。

$$
    f_1 = 2x + 2 
$$
$$
    f_2  = 8.8x + 1
$$
$$
    f_3  = 0.75x - 2
$$

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
## 误差的来源

## 分类问题
包括二分类问题还是多分类问题，我们今天手写数字或文字来进行识别分类，我们人脸检测也属于分类问题。

$$ f(c) = $$
我们需要将物体属性进行数字化，我们可以向量来描述一个物体，然后根据特性来对物体进行区分来达到分类。
### 应用范围

有关分类问题，我们假设一下，有 $C_1$ 和 $C_2 两个类别分类数量为 65 和 78 那么就是
$$ P(C_1) = 65/(78 + 65) =  0.45 $$
$$ P(C_2) = 78/(78 + 65) =  0.55 $$
下面问题就感觉有点绕，我尝试将其说明，假如我们拿到物体 a 那么这个物体类 $C_1$ 的概率是多少

### 条件概率和全概率
![小球问题](https://upload-images.jianshu.io/upload_images/8207483-8f50d6d51b5848b8.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

我们整理一下已知条件,已知条件如下，相比大家看公式就一目了然

$$ P(B_1) = \frac{2}{3} P(Blue | B_1) = \frac{4}{5} P(Green|B_1) = \frac{1}{5}$$
$$ P(B_1) = \frac{1}{3} P(Blue | B_1) = \frac{2}{5} P(Green|B_1) = \frac{3}{5}$$

根据已知条件，我们就可以计算出今天我们取出一个篮球是从 B1 盒子取出的概率
$$ P(B_1 | Blue) = \frac{P(Blue|B_1)P(B_1)}{P(Blue|B_1)P(B_1) + P(Blue|B_2 )P(B_2)}$$

这里根分类是什么关系，暂时还不是那么好理解，我们停下来思考一下，其实分类问题，我们推出物体属于哪一个类别也就是一种概率问题，大家不能否认推出是概率问题，那么我们取出物体可能类别也就转化为概率问题。每一个盒子想象类，取小球动作转化为推出小球是输出哪一个盒子问题

$$ P(C_1 | x) = \frac{P(x | C_1)P(C_1)} {P(x|C_1)P(C_1) + P(x|C_2)P(C_2)} $$

#### 全概率事件
我们得到 $P(C_1|x)$ 就需要，$P(x) = P(x|C_1)P(C_1) + P(x|C_2)P(C_2)$

#### Prior


### 线性回归模型 VS

### 解决方案
#### 

### 图片分类实例
其实有时候做多了也就有感觉，我们通过大量练习返回来思考总结才是自己的
#### 数据集以及数据处理

- 获取数据集
```
fashion_mnist = keras.datasets.fashion_mnist
```
- 训练数据集和测试数据集
```
(train_images,train_labels),(test_images,test_labels) = fashion_mnist.load_data()

```
- 查看数据集结构

```
print(train_images[0].shape)
print(train_labels[0])
```

```
(28, 28)
9
```
```
class_names = ['T恤','裤子','套衫','上衣','外套','凉鞋','衬衫','运动鞋','包','短靴']
```
- 视图查看图片

```
plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.grid(False)
plt.show()
```
![短靴](https://upload-images.jianshu.io/upload_images/8207483-0cb1ace193812cd2.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

```
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i],cmap=plt.cm.binary)
    plt.xlabel(u"\"" + class_names[train_labels[i]] + "\"")
plt.show()
```
![服装分类](https://upload-images.jianshu.io/upload_images/8207483-13aedf9584c7f20f.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

### 网络结构设计
- 定义输入变量
- 设计层
第一层，将 28 * 28 像素矩阵转换为 784 向量作为输入
```
keras.layers.Flatten(input_shape=(28,28)),
```
第二层，
```
keras.layers.Dense(128,activation=tf.nn.relu),
```
输出
```
keras.layers.Dense(10,activation=tf.nn.softmax)

```
- 评估
- 优化

```
model.compile(optimizer="adam",loss="sparse_categorical_crossentropy",metrics=["accuracy"])

```
```
Epoch 1/5
60000/60000 [==============================] - 3s 57us/sample - loss: 0.4979 - acc: 0.8242
Epoch 2/5
60000/60000 [==============================] - 3s 53us/sample - loss: 0.3738 - acc: 0.8658
Epoch 3/5
60000/60000 [==============================] - 3s 52us/sample - loss: 0.3356 - acc: 0.8784
Epoch 4/5
60000/60000 [==============================] - 3s 53us/sample - loss: 0.3112 - acc: 0.8855
Epoch 5/5
60000/60000 [==============================] - 3s 53us/sample - loss: 0.2928 - acc: 0.8929
```
#### 测试
```
print(f"Test Accuracy:{test_acc}")
```
```
# Test Accuracy:0.8672999739646912
```

```
predictions = model.predict(test_images)
print(predictions[0])
print(np.argmax(predictions[0]))
print(test_labels[0])
```

```
cc: 0.8773
[1.2115221e-06 2.8071650e-08 2.4146255e-08 8.1381808e-09 2.2384482e-07
 3.9193379e-03 4.5686298e-07 3.4515742e-02 2.3175874e-06 9.6156067e-01]
9
9
```
## 逻辑线性回归



### 二分类实例
```
imdb = keras.datasets.imdb

(train_data,train_labels),(test_data,test_labels) = imdb.load_data(num_words=10000)

```

```
word_index = imdb.get_word_index()
# key add padding incidate start and indicate unknown and indicate unused
word_index = {k:(v+3) for k,v in word_index.items()}
# consist even 
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2
word_index["<UNUSED>"] = 3

reverse_word_index = dict([(value,key) for (key,value) in word_index.items()])

def decode_review(text):
    return ' '.join([reverse_word_index.get(i,"?") for i in text])

print(train_data[0])
print(decode_review(train_data[0]))
``` 

```
[1, 14, 22, 16, 43, 530, 973, 1622, 1385, 65, 458, 4468, 66, 3941, 4, 173, 36, 256, 5, 25, 100, 43, 838, 112, 50, 670, 2, 9, 35, 480, 284, 5, 150, 4, 172, 112, 167, 2, 336, 385, 39, 4, 172, 4536, 1111, 17, 546, 38, 13, 447, 4, 192, 50, 16, 6, 147, 2025, 19, 14, 22, 4, 1920, 4613, 469, 4, 22, 71, 87, 12, 16, 43, 530, 38, 76, 15, 13, 1247, 4, 22, 17, 515, 17, 12, 16, 626, 18, 2, 5, 62, 386, 12, 8, 316, 8, 106, 5, 4, 2223, 5244, 16, 480, 66, 3785, 33, 4, 130, 12, 16, 38, 619, 5, 25, 124, 51, 36, 135, 48, 25, 1415, 33, 6, 22, 12, 215, 28, 77, 52, 5, 14, 407, 16, 82, 2, 8, 4, 107, 117, 5952, 15, 256, 4, 2, 7, 3766, 5, 723, 36, 71, 43, 530, 476, 26, 400, 317, 46, 7, 4, 2, 1029, 13, 104, 88, 4, 381, 15, 297, 98, 32, 2071, 56, 26, 141, 6, 194, 7486, 18, 4, 226, 22, 21, 134, 476, 26, 480, 5, 144, 30, 5535, 18, 51, 36, 28, 224, 92, 25, 104, 4, 226, 65, 16, 38, 1334, 88, 12, 16, 283, 5, 16, 4472, 113, 103, 32, 15, 16, 5345, 19, 178, 32]
<START> this film was just brilliant casting location scenery story direction everyone's really suited the part they played and you could just imagine being there robert <UNK> is an amazing actor and now the same being director <UNK> father came from the same scottish island as myself so i loved the fact there was a real connection with this film the witty remarks throughout the film were great it was just brilliant so much that i bought the film as soon as it was released for <UNK> and would recommend it to everyone to watch and the fly fishing was amazing really cried at the end it was so sad and you know what they say if you cry at a film it must have been good and this definitely was also <UNK> to the two little boy's that played the <UNK> of norman and paul they were just brilliant children are often left out of the <UNK> list i think because the stars that play them all grown up are such a big profile for the whole film but these children are amazing and should be praised for what they have done don't you think the whole story was so lovely because it was true and was someone's life after all that was shared with us all
```

### 逻辑回归实例
#### 数据
```
(x_train,y_train),(x_test,y_test) = tf.keras.datasets.mnist.load_data()

fig,axes = plt.subplots(1,4,figsize=(7,3))
for img, label, ax in zip(x_train[:4],y_train[:4],axes):
    ax.set_title(label)
    ax.imshow(img)
    ax.axis('off')
plt.show()
```
#### 数据结构
手写数字的库 mnist 已经被用烂了，不过我们还是用这个演示逻辑回归，因为搜集数据的确是一件难事，要不也不会产生专门提供数据用于机器学习训练公司，而且样本价格不菲。
```
print(f'train images: {x_train.shape}')
print(f'train labels: {y_train.shape}')
print(f'test images: {x_test.shape}')
print(f'test labels: {x_test.shape}')
```

```
train images: (60000, 28, 28)
train labels: (60000,)
test images: (10000, 28, 28)
test labels: (10000, 28, 28)
```

#### 数据处理

```
x_train = x_train.reshape(60000,784) / 255
x_test = x_test.reshape(10000,784) / 255

```

```
with tf.Session() as sess:
    y_train = sess.run(tf.one_hot(y_train,10))
    y_test = sess.run(tf.one_hot(y_test,10))

print(y_train[:4])
```

```
[[0. 0. 0. 0. 0. 1. 0. 0. 0. 0.]
 [1. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 1. 0. 0. 0. 0. 0.]
 [0. 1. 0. 0. 0. 0. 0. 0. 0. 0.]]

```
定义参数
```
learning_rate = 0.01
epochs = 50
batch_size = 100
batches = int(x_train.shape[0] / batch_size)

```

$\mathbf{Y} = \sigma(\mathbf{X}\cdot\mathbf{W} + \mathbf{B})$

$
\begin{pmatrix}
    y_1 \\ y_2 \\ \vdots \\ y_{10}
\end{pmatrix}=\sigma\left[
\begin{pmatrix}
    x_1 & x_2 & \cdots & x_{784}
\end{pmatrix}
\begin{pmatrix}
    w_{1,1} & w_{1,2} & \cdots & w_{1,10} \\
    w_{2,1} & w_{2,2} & \cdots & w_{2,10} \\
    \vdots & \vdots & \ddots & \vdots \\
    w_{784,1} & w_{784,2} & \cdots & w_{784,10} \\
\end{pmatrix}
+
\begin{pmatrix}
    b_1 \\ b_2 \\ \vdots \\ b_{10}
\end{pmatrix}
\right]
$
#### 定义参数

```
X = tf.placeholder(tf.float32,[None,784])
Y = tf.placeholder(tf.float32,[None,10])

W = tf.Variable(np.random.randn(784,10).astype(np.float32))
B = tf.Variable(np.random.randn(10).astype(np.float32))
```

```
# prediction
pred = tf.nn.softmax(tf.add(tf.matmul(X,W),B))

# Loss
cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(pred),axis=1))
# 
```
$$ C = \sum -Y\ln(pred)$$
我们看一看损失函数，

```python
x = np.linspace(1/100,1,100)
plt.plot(x,np.log(x))
plt.show()
```
我们来以图形的形式输出一下 log 函数
![图](https://upload-images.jianshu.io/upload_images/8207483-00842e7cff045013.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

在我们通过曲线看出$\ln(x)$ 在1.0之后趋近于 0 

```
a = np.log([[0.04,0.13,0.96,0.12], #correct
        [0.01,0.93,0.06,0.07]]) #incrroet
# labels
b = np.array([[0,0,1,0],
        [1,0,0,0]]) 
```

```
[[0.         0.         0.04082199 0.        ]
 [4.60517019 0.         0.         0.        ]]
```

```
with tf.Session() as sess:
    tf_sum = sess.run(-tf.reduce_mean(a * b,axis=1))
    tf_mean = sess.run(tf.reduce_mean(tf_sum))
```

```
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(epochs):
        for i in range(batches):
            offset = i * epoch
            x = x_train[offset:offset + batch_size]
            y = y_train[offset:offset + batch_size]
            sess.run(optimizer,feed_dict={X:x,Y:y})
            c = sess.run(cost,feed_dict={X:x,Y:y})

        if not epoch % 1:
            print(f'epoch:{epoch} cost={c:.4f}')
```

```
epoch:47 cost=0.6200
epoch:48 cost=0.4541
epoch:49 cost=0.7006
```

```
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(epochs):
        for i in range(batches):
            offset = i * epoch
            x = x_train[offset:offset + batch_size]
            y = y_train[offset:offset + batch_size]
            sess.run(optimizer,feed_dict={X:x,Y:y})
            c = sess.run(cost,feed_dict={X:x,Y:y})

        if not epoch % 1:
            print(f'epoch:{epoch} cost={c:.4f}')
    correct_pred = tf.equal(tf.argmax(pred,1),tf.argmax(Y,1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred,tf.float32))
    acc = accuracy.eval({X:x_test,Y:y_test})
    print(acc)
    ```

```
epoch:0 cost=3.2238
epoch:1 cost=5.1471
epoch:2 cost=3.1629
epoch:3 cost=1.9107
epoch:4 cost=2.1143
epoch:5 cost=2.3709
epoch:6 cost=1.7046
epoch:7 cost=1.2634
epoch:8 cost=2.3583
epoch:9 cost=1.2030
0.7397
```

### 深度神经网络
### 反向传播
#### 梯度下降
$$ \theta = $$
#### 链式法则
$$ y = g(x) z = h(y) $$
$$ \frac{dz}{dx} = \frac{dz}{dy} $$

$$ L(\theta) = \sum_{n=1} C^n(\theta) $$

$$ z = x_1w_1 + x_2w_2 \infinit $$
### CNN
### RNN
### 强化学习


深度神经网有今天的发展不得不要归功于斯坦福华人教授李飞飞的 ImageNet，有了强大数据集其中包含 1400 万张图，22000 种分类，其中多达 100 万张图片中物体是有边框。AlexNet 是 2012 分类冠军 top5 出错率为 16.4%
在 AlexNet 中有一些突破

### 
- 越复杂的模型

### 梯度消失
softmax = tf.exp(logits) / tf.reduce_
可以去文档 API 可以去官方的文档去查

### 卷积神经网络
- 神经网络入门
- 反向传播
- 梯度下降优化方法
### 卷积神经网
- 卷积、池化、全连接
- 全卷积神经网络
```
[Train] Step: 9499, loss: 1.07731, acc: 0.60000
[Train] Step: 9999, loss: 0.91756, acc: 0.75000
[Test ] Step: 10000, acc: 0.69900
```
### 卷积神经网络进阶
#### 模型演变
- AlexNet
- VGG
- ResNet
- Inception
- MobileNet
#### 对比
#### 多种网络结构
- 不用问题有不同解决方法
- 不同网络结构的技巧不同
- 不同网络结果应用环境不同
进化从基本神经网络到更宽的神经网络
优化组合 Inception + Res = InceptionResNet
NASNet 强化学习和卷积神经网络
移动端的模型 MobileNet
### AlexNet


2012 年我在干什么，8 层
#### AlexNet 特点 网络结构，有分离有交叉
- 好处并行化
- Dropout 随机丢弃（为什么用在全连接层）
- 4096 * 4096 16m 全连接参数过多样本过多，过拟合在测试集表现不好，也就是泛化能力差。
#### dropout 的原理
- 组合解释（dropout的原理）相对全连接是子网络，最后可以看出是多个子网络组合结果得到网络
- 动机解释 消除神经元之间的依赖、增强泛化能力
- 数据解释 相当于数据增强，随机采样，也相当于从不同角度来看图片。
- 经验参数
    - Dropout 0.5
    - Batch_size 128
    - SGD momentum = 0.9
    - Learning rate = 0.01 一定次数降低为 1/10
    - 7个CNN做 ensemble 18.2% -> 15.5%

## VGGNet(11-19层)
在 2014 年，由于 AlexNet 的出现，引起大家都 CNN 的关注，所以牛津大学计算机视觉组（Visual Geometry Group）和 Google DeepMind 公司的研究员一起研发出了新的深度卷积神经网络 —— VGGNet，并取得了ILSVRC2014比赛分类项目的第二名（第一名是GoogLeNet，也是同年提出的）和定位项目的第一名。
深度神经网络的深度可以做的更深也得益于 GPU 出现。


### 网络结构
    - VGG由5层卷积层
    - 3层全连接层
    - softmax输出层构成
    - 层与层之间使用max-pooling（最大化池）分开，所有隐层的激活单元都采用ReLU函数。
### 特点
    我们知道 CNN 一般都包含卷积层、池化层和全连接层，我们按这些层分别说明一下 VGGNet 的特点    
    - 卷积层
        - 在 VGGNet 中尽量多使用 3 x 3 的卷积核来带了更多特征，好处减少参数数量，增加线性。
        - 使用 2 个 3 * 3 的卷积层来代替 5 * 5 卷积层 
        - 使用 3 个 3 * 3 的卷积层来代替 7 * 7 卷积层
    - 池化层
        - 在池化层使用（2,2）的步长处理池化层，为什么用（2,2）表示更小池化层会得到更多信息。
    - 1 * 1 卷积层看做非线性变换
    - 每经过一个 pooling，通道数目翻倍
    - 视野区域
        - 2 个 3 * 3 的卷积层= 5 * 5 
        - 2 层比 1 层更多一次非线性变换
        - 参数减少 28% 5 * 5 * channel * 3 * 3 + 3 * 3

- 递进式训练
    - 先训练浅层神经网然后再训练深层网络结构
```
[Train] Step: 9499, loss: 0.98760, acc: 0.55000
[Train] Step: 9999, loss: 1.12063, acc: 0.70000
[Test ] Step: 10000, acc: 0.73300
```
在跌跌撞撞搞了一段时间机器学习，感觉有必要总结一些
### 评估
- 模型：就是一些列函数估计

我们任务就找到最佳 f(x)，也就是我们要训练模型的终极目标，我就是要找到这个最佳f(x)
$$ \hat{y} = \hat{f}(x) $$
不过我们尝试f(x) 与目标总会有一些差距，今天我们就来研究是什么导致这个差距，然后尝试优化这些因素来实现缩小与最终目标距离。
- 偏离(Baise) 表示偏离目标值
- 离散 (Variance) 表示我们训练集的值离散程度
我们通过手上训练集来找到f(x),其实我们每次找到都是估计f(x)
$$ m = \frac{1}{N}\sum_{n}x_n$$
$$ Var[m] = \sigma^2$$
$$ s^2 = $$

$$ E[f^*] = \overline{f}$$

简单model
$$ y = b + w \cdot x_i $$
$$ y = b + w_1 \cdot x_i + w_2 \cdot (x_i)^2$$

#### 偏离值
$$ E[f^*] = \overline{f} $$
如果一个比较简单模型可能会有较大偏差，而复杂模型会

#### 过拟合和欠拟合
- 过拟合(overfitting)
在训练集上得到好的效果，而在测试集上，
- 欠拟合（Underfitting）
需要从新设计模型，考虑一些更多特征，这是收集更多数据

- 消除离散
    - 可以增加数据，收集数据比较难，可以通过选择图片，对图片进行变换来获取更多数据。
    - 通过（regularization)可能会损失偏离值
- 减少偏差
- 需要重新设计模型

#### 模型的选择
- there is trade-off between bias and variance


### 梯度下降(Gradient Descent)
$$ \theta^* = arg min L(\theta)$$

$$ \theta = \ $$

$$ \partial L(\theta_1^0) $$

### 有关调整学习率(learning rate）
合适
学习率如果较小，步伐过小我们就无法忍受，
参数变化来查看
Loss 和 参数更新的关系反应learning rate，在做线性回归时候我们需要查看，learning rate 是我们可以调整的。
#### 自动调整学习率，
通常做法随着参数调整学习率会越来越小。
$$ \eta $$
#### Adagrad
$$ w^{t+1} \leftarrow w^t - \eta^t g^t$$
$$$$
### 反向传播(Backpropagation)
$$ w_a^+ = w_1 + \eta \frac{\partial E_{total}}{\partial w_1}$$


### ResNet 
是 2015 年分类比赛冠军，AlphaGo 最初基础神经网络就是，何开明等人研究开发，ResNet 明显的进步就是可以让神经网络有相当深的深度，可以实现从 32 - 152 层神经网络。
#### 网络深度好处
- 网络越深所能表达特征越高，更深网络具有更强大表达能力
#### 加深层次带来问题
- 梯度弥散
$$ w_a^+ = w_1 + \eta \frac{\partial E_{total}}{\partial w_1}$$
$$ y_i = \sigma(z_i) = \sigma(w_ix_i + b_i)$$
$$ \frac{\partial E_total}{\partial w_1} = \frac{\partial E_total}{\partial y_4}$$
通常 w_i 在 0 - 1 附近如果 w_i 

- 浅层网络和深层网络
假设我们浅层网络已经找到最优 f(x),但实际情况附加层（后添加的）的网络层并不训练为恒等，导致浅层和深层不具备相同误差。
- 退化问题
层数过深的平原网络具有更高的训练误差
- 多层平原网路
针对**退化问题**和**梯度弥散**问题，退出残差网络

- 即使加深网络结果，
- 深层网络难于优化，而不是深层网络不好，深层网络至少可以学习到浅层神经网的
- y=x 虽然增加了深度、单误差不会增加
- 模型结构
    - 残差学习
    - 恒等变换，恒等变化子结构，F(x) + X
    - 残差网路
    - 没有全连接层，从而减少了参数，强调卷积层弱化全连接层
- ResNet 家族    
    - ResNet-34 
    - ResNet-101

    - 先用一个普通卷积层，stride=2
    - 再经过一个 3 * 3 的 max_pooling 层

```
[Train] Step: 499, loss: 1.41713, acc: 0.35000
```

### MobileNet
```
[Train] Step: 499, loss: 1.57021, acc: 0.45000
[Train] Step: 499, loss: 1.57021, acc: 0.45000
[Train] Step: 999, loss: 1.37653, acc: 0.50000
[Test ] Step: 1000, acc: 0.46700
[Train] Step: 1499, loss: 1.37062, acc: 0.50000
[Train] Step: 1999, loss: 1.22970, acc: 0.65000
[Test ] Step: 2000, acc: 0.51400
[Train] Step: 2499, loss: 1.42647, acc: 0.5500
```

2012 AlexNet
2014 VGGNet
2015 ResNet
InceptionV1-V4
MobileNet 深度可分离卷积

### 卷积神经网络调参
#### 更多优化算法
##### 随机梯度下降
    - 局部极值
    - Saddle point 问题
##### 动量梯度下降
##### 现存问题
    - 受到初始学习率影响很大
    - 每一个维度的学习率一样(全局设置学习率）
    - 很难学习到稀疏数据，多数情况是得不到梯度更新
##### 解决方案
###### AdaGrad 算法
调整学习率
$$ n_t = nt_t + g_t^2 $$
$$ \Delta\theta_t = - \frac{\eta}{\sqrt{n_t + \epsilon}}  $$
```python
grad_squared = 0
with True:
    dx = compute_gradient(x)
    grad_squared += dx * dx
    x -= learning_rate * dx / (np.sqrt(grad_squared) + 1e-7)

```
####### 算法特点
- 前期 regularizer 较小，放大梯度
- 后期 regularizer 较大，缩小梯度
- 梯度随训练次数降低
- 每个分量有不同的学习率
####### 缺点
- 学习率设置过大，导致 regularizer 影响过于敏感
- 后期，regularizer 累积值太大，体现

##### RMSProp
- Adagrad 的变种
- 由累积平方梯度变为平均平方梯度
- 解决了后期提前结束的问题
```python
grad_squared += decay_rate * grade_squared + (1 - decay_rate) * dx * dx
```
##### 自定义学习率
- 可以自定义随指数衰减的学习率
$$ \alpha = \alpha_0e^{-kt} $$
    - k 是系数
    - t 迭代次数
$$  $$
- 对于稀疏数据（），使用学习率可（高纬度小数据
- SGD 通常训练时间较长，最终效果比较好，但是需要好的初始化和学习率
- 需要训练较深复杂网络且需要快速收敛，推荐 adam
- Adagrad RMSprop Adam 是比较相近的算法，表现差异不大，Adam 相对稳定
### 激活函数
#### sigmoid
- 输入非常大或非常小时没有梯度
- 在（-6,6）之间变化较大
- 输出均值为 0.5
- Exp计算比较复杂
- 梯度消失
$$ \sigma(x) = \frac{1}{1+e^{-1}} $$
- 网络初始化
- 归一化
- 数据增强
- 更多调参方法


## SVM
### 再次回到二分类问题
$$
g(x) =
\begin{cases}
f(x) > 0 output =+1 \\
f(x) < 0 output =-1
\end{cases}
$$
在看一看我们训练数据集样子
$$
x^1 \hat{y}^1
$$

## Keras
### 卷积神经网络
#### 数据集
下载 window 的 Cat 和 Dog 数据集作为训练数据集
我们可以先列出对于数据处理几个步骤
- 下载数据集
- 拆分为训练数据集和测试数据集
- 分析（查看）数据
- 处理数据来达到适合训练的数据结构
    - 包括灰度处理
    - 对图片缩放
    - 整理标签为 one-hot 格式数据等等

```
DATADIR = "PetImages"
CATEGORIES = ['Dog','Cat']

for category in CATEGORIES:
    print(category)
    path = os.path.join(DATADIR,category)
    for img in os.listdir(path):
        img_array = cv2.imread(os.path.join(path,img),cv2.IMREAD_GRAYSCALE)
        plt.imshow(img_array,cmap="gray")
        plt.show()
        break
    break
```
![狗](https://upload-images.jianshu.io/upload_images/8207483-02007a6a3603a8c0.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
```
IMG_SIZE = 50
print(img_array.shape)
new_array = cv2.resize(img_array,(IMG_SIZE,IMG_SIZE))
plt.imshow(new_array,cmap='gray')
plt.show()
```
![50_50_狗)](https://upload-images.jianshu.io/upload_images/8207483-1cfa2b75e78c04f3.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

```

DATADIR = "PetImages"
CATEGORIES = ['Dog','Cat']
IMG_SIZE = 50
training_data = []
def create_training_data():
    for category in CATEGORIES:
        # print(category)
        path = os.path.join(DATADIR,category) # path to cats or dogs dir
        # print(path)
        class_num = CATEGORIES.index(category)
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path,img),cv2.IMREAD_GRAYSCALE)
                new_array = cv2.resize(img_array,(IMG_SIZE,IMG_SIZE))
                training_data.append([new_array,class_num])
            except Exception as e:
                pass
```

```
24946
```
通过获取数据我们先读取 Dog 然后读取 Cat 这样就找出机器在读取数据时候先读取（看到）都是 Dog 数据，这样只要仔细一想就知道不适合机器学习。
```
random.shuffle(training_data)
```
```
for sample in training_data[:10]:
    print(sample[1])
```
```
1
1
1
0
1
1
0
1
0
1
```
```
X = np.array(X).reshape(-1,IMG_SIZE,IMG_SIZE,1)
```
每一次


```
pickle_out = open("X.pickle","wb")
pickle.dump(X, pickle_out)
pickle_out.close()

pickle_out = open("y.pickle","wb")
pickle.dump(y, pickle_out)
pickle_out.close()

pickle_in = open("X.pickle","rb")
X = pickle.load(pickle_in)
print(X[1])
```
```
Epoch 1/3
22451/22451 [==============================] - 77s 3ms/sample - loss: 0.6500 - acc: 0.6149 - val_loss: 0.6271 - val_acc: 0.6397
Epoch 2/3
22451/22451 [==============================] - 76s 3ms/sample - loss: 0.5349 - acc: 0.7360 - val_loss: 0.5465 - val_acc: 0.7255
Epoch 3/3
22451/22451 [==============================] - 75s 3ms/sample - loss: 0.4796 - acc: 0.7738 - val_loss: 0.4885 - val_acc: 0.7671
```