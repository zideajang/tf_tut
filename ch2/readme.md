##（7月2日）
### 什么是卷积
卷积是一种数学运算，这种运算本身很复杂，目的简化更复杂的数据表达，过滤掉数据中噪声提取出关键的特征，所以卷积运算也被称为过滤，在计算机视觉中也应用广泛。
![卷积操作示意图](https://upload-images.jianshu.io/upload_images/8207483-4088db2e29f35319.jpg?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

 
$$
 \left[
 \begin{matrix}
   0 & 0 & 0 \\
   0 & 1 & 1 \\
   0 & 1 & 2
  \end{matrix}
  \right] 
  
  \cdot

 \left[
 \begin{matrix}
   4 & 0 & 0 \\
   0 & 0 & 0 \\
   0 &  & -4
  \end{matrix}
  \right] 
= -8
  
$$

具体计算过程在图右上角可见。

### 卷积神经网络基础
#### 局部感知野
#### 共享参数
#### 多卷积核
我们需要对一张图片提取多个特征，不同卷积核用于提取不同特征，
#### 池化
### 训练
#### 池化层反向传播
我们之前已经学习到池化层有两种类型分别是**平均池化**和**最大池化**
- 平均池化层的残差传播
输入是一个 4 x 4 矩阵池化后就是 2 x 2 矩阵，然后我们反向时候保持总和不变将每一值均摊就会
$$
 \left[
 \begin{matrix}
   1 & 2  \\
   6 & 3 
  \end{matrix}
  \right] 
  $$

  $$
 \left[
 \begin{matrix}
   0.25 & 0.25 & 1 & 1  \\
   0.25 & 0.25 & 1 & 1  \\
   1.5 & 1.5 & 0.75 & 0.75 \\ 
   1.5 & 1.5 & 0.75 & 0.75  
  \end{matrix}
  \right]
  $$
- 最大池化层的残差传播
$$
 \left[
 \begin{matrix}
   1 & 2  \\
   6 & 3 
  \end{matrix}
  \right] 
  $$

  $$
 \left[
 \begin{matrix}
   1 & 0 & 0 & 0  \\
   0 & 0 & 2 & 0  \\
   0 & 0 & 0 & 3 \\ 
   0 & 6 & 0 & 0  
  \end{matrix}
  \right]
  $$

#### 卷积层反向传播
- 卷积前的矩阵
$$
 \left[
 \begin{matrix}
   x_{00} & x_{01} & x_{02}   \\
   x_{11} & x_{11} & x_{12}   \\
   x_{22} & x_{21} & x_{22}   
  \end{matrix}
  \right]
$$
- 卷积核矩阵
$$
 \left[
 \begin{matrix}
   k_{00} & k_{01}    \\
   k_{12} & k_{11}    
  \end{matrix}
  \right]
$$

- 卷积之后矩阵
$$
 \left[
 \begin{matrix}
   y_{00} & y_{01}    \\
   y_{12} & y_{11}    
  \end{matrix}
  \right]
$$

- 卷积后的残差矩阵
$$
 \left[
 \begin{matrix}
   \delta^{l+1}_{00} & \delta^{l+1}_{01}    \\
   \delta^{l+1}_{12} & \delta^{l+1}_{11}    
  \end{matrix}
  \right]
$$

- 卷积

### AlexNet
**AlexNet**
### ResNet
是由微软 kaiming He 等 4 名华人提出