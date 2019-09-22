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
### ResNet 
是 2015 年分类比赛冠军
- 加深层次问题
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