### 章节名称
#### 节名称
##### 小节名称



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

$$ w^1 \leftarrow w^0 - \eta\frac{dL}{dw} | _{w=w^0}$$
$$ w^2 \leftarrow w^1 - \eta\frac{dL}{dw} | _{w=w^1}$$
当切线为负，我们就增加 w 值像右移动 $\eta$ 是学习速度，也就是移动步长。知道走到微分为 0 时候我们停止前进，所以我们就会思考梯度下降不一定会知道最小值。

上面考虑是一个参数情况，接下来我们来考虑 2 参数
$$ \frac{\partial L}{\partial w} | _{w=w^0,b=b^0},\frac{\partial L}{\partial b} | _{w=w^0,b=b^0} $$

### 线性回归实例


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

#### 优化
```
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)


```

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




## 分类问题
## 逻辑线性回归


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