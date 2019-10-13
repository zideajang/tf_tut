import numpy as np
import tensorflow as tf
from ops import *
import matplotlib.pyplot as plt 
import os #创建输出文件
from generator import Generator
from discriminator import Discriminator

class DCGAN:
    def __init__(self, img_shape,sample_folder_name, iterations=15000, lr_gen=0.0001,lr_dc = 0.0003, z_shape=100,batch_size=64,beta1=0.5,sample_interval=500):
        # lr_gen = 学习率
        # lr_dc 学习率判别器
        # z_shape 生成输入的形状
        # batch_size 较大值减慢训练速度 较小值
        # sample_interval 创建图片的间隔
        pass
    def train(self):
        pass
    def generate_sample(self,iteration):
        pass

if __name__ == '__main__':
    pass