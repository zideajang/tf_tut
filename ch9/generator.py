#coding=utf-8
import numpy as np
import tensorflow as tf
from ops import *
from tensorflow.layers import batch_normalization
from tensorflow.keras.layers import UpSampling2D

class Generator:
    def __init__(self):
        layer_sizes = [512,256,128,1]
        with tf.variable_scope('g'):
            print("初始化生成器权重")
            self.W1 = init_weights([100, 7*7*layer_sizes[0]]])
            self.W2 = init_weights([3, 3, layer_sizes[0]], layer_sizes[1]])
            self.W3 = init_weights([3, 3, layer_sizes[1], layer_sizes[2]])
            self.W4 = init_weights([3, 3, layer_sizes[2], layer_sizes[3]])

        pass
    def forward(self,X,momentum=0.5):
        pass
    