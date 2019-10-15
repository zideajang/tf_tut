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

# 生成器实现
def conv2d_transpose(inputs,out_channel,name,training,with_bn_relu = True):
    with tf.variable_scope(name):
        conv2d_trans = tf.layers.conv2d_transpose(inputs,out_channel,[5,5],strides=(2,2),padding='SAME')

        if with_bn_relu:
            bn = tf.layers.batch_normalization(conv2d_trans,training=training)
            return tf.nn.relu(bn)
        else:
            return conv2d_trans
# 定义卷积层生成器
def conv2d(inputs, out_channel, name, training):
    # 在判别器中使用
    def leaky_relu(x, leak = 0.2, name=''):
        return tf.maximum(x, x * leak,name=name)
    with tf.variable_scope(name):
        conv2d_output = tf.layers.conv2d(inputs,out_channel,[5,5],strides=(2,2),padding='SAME')

        bn = tf.layers.batch_normalization(conv2d_output,training=training)
        return leaky_relu(bn,name='outputs')
        

class Generator(object):
    def __init__(self,channels,init_conv_size):
        self._channels = channels
        self._init_conv_size = init_conv_size
        self._reuse = False
    def __call__(self,inputs,training):
        # 将输入变为 tensorflow 的 tensor 格式数据
        inputs = tf.convert_to_tensor(inputs)
        # 定义命名空间
        with tf.variable_scope('generator',reuse=self._reuse):
            """
                random_vector -> fc -> self._channels[0] * init_conv_size ** 2
                -> reshape -> [init_conv_size,init_conv_size,channels[0]]
            """
            # 通过反卷积将随机向量还原为图片，定义新的作用来完成此任务
            with tf.variable_scope('inputs_conv'):
                fc = tf.layers.dense(
                    inputs,
                    self._channels[0] * self._init_conv_size * self._init_conv_size)
                # 进行矩阵变形，这里还有 batch_size 维度将其设置为 -1
                conv0 = tf.reshape(fc,[-1,self._init_conv_size,self._init_conv_size,self._channels[0]])
                # 
                bn0 = tf.layers.batch_normalization(conv0,training=training)
                # 非线性变换
                relu0 = tf.nn.relu(bn0)
            # 定义一个变量来
            deconv_inputs = relu0
            # 因为channels 第一个已经用了
            for i in range(1,len(self._channels)):
                # 判断是否为最后一层，如果是最后一层需要做
                with_bn_relu = (i != len(self._channels) - 1)
                # 调用之前定义好的函数来进行反卷积处理
                deconv_inputs = conv2d_transpose(
                    deconv_inputs,
                    self._channels[i],
                    "deconv-%d" % i, #
                    training,
                    with_bn_relu)
                # 返回输入赋值给 deconv_inputs
            img_inputs = deconv_inputs
            with tf.variable_scope('generate_imgs'):
                # imgs 值在 [-1,1] 之间
                imgs = tf.tanh(img_inputs,name='imgs')
        # 进行复用                    
        self._reuse = True            
        self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope='generator')
        return imgs
# 定义判别器
class Discriminator(object):
    # 初始化函数
    def __init__(self,channels):
        self._channels = channels
        self._resue = False
    # call 函数
    def __call__(self,inputs, training):
        # input 输入和是否为训练集，首先将输入变为tensor
        inputs = tf.convert_to_tensor(inputs,dtype=tf.float32)

        conv_inputs = inputs
        with tf.variable_scope('discriminator',reuse=self._resue):
            for i in range(len(self._channels)):
                # 每一步都做卷积操作
                conv_inputs = conv2d(conv_inputs,
                                    self._channels[i],
                                    'conv-%d' % i,
                                    training)
            # flatten
            fc_inputs = conv_inputs
            with tf.variable_scope('fc'):
                flatten = tf.layers.flatten(fc_inputs)
                logits = tf.layers.dense(flatten,2,name='logits')
        self._resue = True
        self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope='discriminator')
        return logits
class DCGAN(object):
    def __init__(self,hps):
        g_channels = hps.g_channels
        d_channels = hps.d_channels

        self._batch_size = hps.batch_size
        self._init_conv_size = hps.init_conv_size
        self._z_dim = hps.z_dim
        self._img_size = hps.img_size

        self._generator = Generator(g_channels,self._init_conv_size)
        self._discriminator = Discriminator(d_channels)
    
    def build(self):
        """ 构建计算图 """
        self._z_placeholder = tf.placeholder(tf.float32,(self._batch_size,self._z_dim))
        self._img_placeholder = tf.placeholder(tf.float32,(self._batch_size,self._img_size,self._img_size,1))

        generated_imgs = self._generator(self._z_placeholder,training=True)

        f_img_logits = self._discriminator(generated_imgs,training = True)
        r_img_logits = self._discriminator(self._img_placeholder,training = True)


        loss_on_f_to_r = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.ones([self._batch_size],dtype=tf.int64),logits=f_img_logits))
        loss_on_f_to_f = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.zeros([self._batch_size],dtype=tf.int64),logits=f_img_logits))

        loss_on_r_to_r = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.zeros([self._batch_size],dtype=tf.int64),logits=r_img_logits))

        tf.add_to_collection('g_losses',loss_on_f_to_r)
        tf.add_to_collection('d_losses',loss_on_f_to_f)
        tf.add_to_collection('d_losses',loss_on_r_to_r)

        loss = {
            'g':tf.add_n(tf.get_collection('g_losses'),name='total_g_loss'),
            'd':tf.add_n(tf.get_collection('d_losses'),name='total_d_loss')
        }
        return (self._z_placeholder, self._img_placeholder,generated_imgs,loss)
    def build_train_op(self,losses,learning_rate,beta1):
        """
        """
        g_opt = tf.train.AdamOptimizer(learning_rate=learning_rate,beta1=beta1)
        
        d_opt = tf.train.AdamOptimizer(learning_rate=learning_rate,beta1=beta1)

        g_opt_op = g_opt.minimize(losses['g'],var_list=self._generator.variables)
        d_opt_op = d_opt.minimize(losses['d'],var_list=self._discriminator.variables)

        with tf.control_dependencies([g_opt_op,d_opt_op]):
            return tf.no_op(name='train')

dcgan = DCGAN(hps)
z_placeholder,img_placeholder,generated_imgs,losses = dcgan.build()

train_op = dcgan.build_train_op(losses,hps.learning_rate,hps.beta1)

def combine_imgs(batch_imgs, img_size, rows= 8, cols=16):
    """ Combines all images in a batch into a big pic """
    # batch_imgs: [batch_size, img_size, img_size, 1]
    result_big_img = []
    for i in range(rows):
        row_imgs = []
        for j in range(cols):
            # [img_size, img_size,1]
            img = batch_imgs[cols * i + j]
            img = img.reshape((img_size,img_size))
            img = (img + 1) * 127.5
            row_imgs.append(img)
        row_imgs = np.hstack(row_imgs)
        result_big_img.append(row_imgs)
    # [8 * 32, 16 * 32]
    result_big_img = np.vstack(result_big_img)
    result_big_img = np.asarray(result_big_img,np.uint8)
    result_big_img = Image.fromarray(result_big_img)
    return result_big_img

init_op = tf.global_variables_initializer()
train_steps =  10000
with tf.Session() as sess:
    sess.run(init_op)
    for step in range(train_steps):
        batch_imgs, batch_z = mnist_data.next_batch(hps.batch_size)
        fetches = [train_op,losses['g'],losses['d']]
        should_sample = (step + 1) % 50 == 0
        if should_sample:
            fetches += [generated_imgs]
        output_values = sess.run(fetches,feed_dict={z_placeholder:batch_z,img_placeholder:batch_imgs})

        _, g_loss_val, d_loss_val = output_values[0:3]
        logging.info('step: 4%d, g_loss:4%.3f, d_loss:%4.3f' % (step,g_loss_val,d_loss_val))

        if should_sample:
            gen_imgs_val = output_values[3]
            gen_img_path = os.path.join(output_dir,'%05d-gen.jpg' % (step + 1))
            gt_img_path = os.path.join(output_dir,'%05d-gt.jpg' % (step + 1))

            gen_img = combine_imgs(gen_imgs_val,hps.img_size)
            gt_img = combine_imgs(batch_imgs,hps.img_size)

            gen_img.save(gen_img_path)
            gt_img.save(gt_img_path)