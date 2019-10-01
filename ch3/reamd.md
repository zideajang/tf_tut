## 优化

### 激活函数

### 初始化

### 数据增强
- 归一化
- 图像变换
    - 翻转
    - 拉伸
    - 裁剪
    - 变形
- 颜色变换
    - 对比度
    - 亮度
- 多尺度

### 调参技巧
- 更多数据
- 神经网络添加层次
- 关注新技术、新模型
- 增大训练的迭代次数
- 尝试正则化
- 使用更多的 GPU 来加速训练

### 可视化
- 损失
- 梯度
- 准确率
- 学习率

### 附加调参技巧

- 在标准数据集上训练
- 在小数据集上过拟合
- 数据集分布平衡
- 使用预调整好的稳定模型结构
- Fine-tuning 预训练的网络结构上进行微调

### 可视化
1. 指定面板上显示的变量
2. 训练过程中将这些变量计算出来，输出到文件中
3. 文件解析，tensorboard 可以解析文件 --logdir=dir.

为损失函数 loss 和准确函数建立总结 summary

```
# visilization loss and accuracy
loss_summary = tf.summary.scalar('loss',loss)
# 'loss' <10,1.1> <20 1.08
accuracy_summary = tf.summary.scalar('accuracy',accuracy)
```

```
source_image = (x_image + 1) * 127.5
inputs_summary = tf.summary.image('inputs_image',source_image)

```
```
merged_summary = tf.summary.merge_all()

merged_summary_test = tf.summary.merge([loss_summary,accuracy_summary])

```
