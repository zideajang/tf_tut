#coding=utf-8

# 预先定义所有操作
import numpy as np
import tensorflow as tf

# 初始化权重
def init_weights(shape):
    return tf.Variable(tf.random_normal(shape=shape,stddev=0.02)) # Stddev can be changed
# 初始化偏移值
def init_bias(shape):
    return tf.Variable(tf.zeros(shape)) #偏移值初始都给了 0 

# 创建卷积层
def conv2d(x, filter, strides, padding):
    return tf.nn.conv2d(x, filter, strides=strides,padding=padding)

# 定义价值函数（也叫损失函数)
def cost(labels, logits):
    # 使用交叉熵
    return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=labels,logits=logits))

# Sigmoid 创建输出为 0 和 1 == Fake 或 Real

