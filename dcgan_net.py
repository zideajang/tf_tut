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
# print hps.img_size
# print hps.g_channels
# print mnist.train.images.shape

# 提供数据
class MnistData(object):
    # mnist_train 训练集图片
    # z_dim 和 img_size 图片大小需要相等
    # 输出图片大小
    def __init__(self, mnist_train, z_dim, img_size):
        self._data = mnist_train
        self._example_num = len(self._data)
        self._z_data = np.random.standard_normal((self._example_num,z_dim))

        self._indicator = 0
        self._resize_mnist_img(img_size)
        self._random_shuffle()

    def _random_shuffle(self):
        p = np.random.permutation(self._example_num)
        self._z_data = self._z_data[p]
        self._data = self._data[p]
    
    def _resize_mnist_img(self,img_size):
        """
            对mnist 图片进行缩放，先将 numpy 转为 PIL 图片对象
            然后将再将 PIL 图片对象转换为 numpy 对象
        """
        data = np.asarray(self._data * 255, np.uint8)
        # [example_num,784] -> [exmaple_num,28,28]
        data = data.reshape((self._example_num,28,28))
        new_data = []
        for i in range(self._example_num):
            img = data[i]
            img = Image.fromarray(img)
            img = img.resize((img_size,img_size))
            img = np.asarray(img)
            img = img.reshape((img_size,img_size,1))
            new_data.append(img)
        new_data = np.asarray(new_data,dtype=np.float32)
        new_data = new_data / 127.5 - 1
        # slef._data:[num_example,img_size,img_size]
        self._data = new_data
    
    def next_batch(self,batch_size):
        end_indicator = self._indicator + batch_size
        if end_indicator > self._example_num:
            self._random_shuffle()
            self._indicator = 0
            end_indicator = self._indicator + batch_size
        assert end_indicator < self._example_num

        batch_data = self._data[self._indicator:end_indicator]
        batch_z = self._z_data[self._indicator:end_indicator]
        self._indicator = end_indicator
        return batch_data,batch_z
    
mnist_data = MnistData(mnist.train.images,hps.z_dim,hps.img_size)
batch_data,batch_z = mnist_data.next_batch(5)
# print batch_data
# print batch_data[0][16]
# print batch_z