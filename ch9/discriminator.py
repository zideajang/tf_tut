#coding=utf-8
import numpy as np
import tensorflow as tf

from ops import *
from tensorflow.layers import batch_normalization

class Discriminator:
    def __init__(self,img_shape):
        _,_,channels = img_shape
        # 初始化权重和偏移值
        # 定义 Variable Scope(变量作用域),便于区分生成器和判别器
        layer_sizes = [64,64,128,256]
        with tf.variable_scope('d'):
            print("初始化判别器权重")

            self.W1 = init_weights([5,5,channels,layer_sizes[0]])
            self.b1 = init_bias([layer_sizes[0]])

            self.W2 = init_weights([3,3,layer_sizes[0],layer_sizes[1]])
            self.b2 = init_bias([layer_sizes[1]])
            
            self.W3 = init_weights([5,5,layer_sizes[1],layer_sizes[2]])
            self.b3 = init_bias([layer_sizes[2]])
            
            self.W4 = init_weights([5,5,layer_sizes[2],layer_sizes[3]])
            self.b4 = init_bias([layer_sizes[3]])
            
            self.W5 = init_weights([5,5, 7*7*layer_sizes[3],1])
            self.b5 = init_bias([1])

    def forword(self,X,momentum=0.5):
        # 创建前向传播
        # 4 个卷积层而且没有使用池化层，通过在卷积层上使用步长为 2 来达到全连接层的效果（减小图片尺寸)
        # 1 个全连接层

        # 定义第一层
        # 步长形状为 [batch,height,width,channels]
        z = conv2d(X, self.W1, [1, 2, 2, 1], padding="SAME") # 14x14x64
        # 定义偏移值
        z = tf.nn.bias_add(z, self.b1)
        # 激活函数 这里使用 leak
        z = tf.nn.leaky_relu(z)
        
        # 定义第二层
        z = conv2d(z, self.W2, [1, 1, 1, 1], padding="SAME")  # 14x14x64
        z = tf.nn.bias_add(z, self.b2)
        # 归一化冲量为 0.5 
        z = batch_normalization(z, momentum=momentum)
        z = tf.nn.leaky_relu(z)

        # 定义第三层
        z = conv2d(z, self.W3, [1, 2, 2, 1], padding="SAME")  # 7x7x128
        z = tf.nn.bias_add(z, self.b3)
        z = batch_normalization(z, momentum=momentum)
        z = tf.nn.leaky_relu(z)

        z = conv2d(z, self.W4, [1, 1, 1, 1], padding="SAME") # 7x7x256
        z = tf.nn.bias_add(z, self.b4)
        z = batch_normalization(z, momentum=momentum)
        z = tf.nn.leaky_relu(z)

        # 全连接层
        # 通过 flatten image
        z = tf.reshape(z, [-1, 7*7*256])
        logits = tf.matmul(z, self.W5)
        logits = tf.nn.bias_add(logits, self.b5)
        return logits