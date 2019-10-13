#coding=utf-8
"""
1. 提供数据（图像数据和随机向量)
2. 计算图的构建
    生成器(generator)和判断器(discriminator)
4. 训练
"""
import os
import sys
import tensorflow as tf
from tensorflow import logging
from tensorflow import gfile
import pprint
import cloudpickle
import numpy as np
import random
import math
from PIL import Image
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data/',one_hot=True)
output_dir = './local_run'
if not gfile.Exists(output_dir):
    gfile.MakeDirs(output_dir)

def get_default_params():
    return tf.contrib.training.HParams(
        z_dim = 100, #随机向量的长度
        init_conv_size = 4, #将一个向量变为矩阵，控制将向量特征初始的大小
        g_channels = [128,64,32,1], #
        d_channels = [32,64,128,256], # 步长为 2 特征图减半通道数量加倍
        batch_size = 128,
        learning_rate = 0.002,
        beta1 = 0.5,
        img_size = 32 #要生成图像的大小，图像都是正方形所以长和宽都是相等的
    )
hps = get_default_params()
print hps.img_size
print hps.g_channels
print mnist.train.images.shape