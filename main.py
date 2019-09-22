import tensorflow as tf
# print(tf.__version__)
import numpy as np
import cPickle
import os
import utils
from utils import CifarData
# save file format as numpy
# file format as cPickle
CIFAR_DIR = "./cifar-10-batches"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# print(os.listdir(CIFAR_DIR))

train_filenames = [os.path.join(CIFAR_DIR, 'data_batch_%d' % i)
                   for i in range(1, 6)]
test_filenames = [os.path.join(CIFAR_DIR, 'test_batch')]
train_data = CifarData(train_filenames, True)
test_data = CifarData(test_filenames, False)

# ['data_batch_1', 'readme.html', 'batches.meta', 'data_batch_2', 'data_batch_5', 'test_batch', 'data_batch_4', 'data_batch_3']

# [None]
# define input and output placeholder recieved input 
# None present confirm count of smaples
x = tf.placeholder(tf.float32,[None,3072])
y = tf.placeholder(tf.int64,[None])
# (3071,1)
w = tf.get_variable('w',[x.get_shape()[-1],1],initializer=tf.random_normal_initializer(0,1))
# (1,)
b = tf.get_variable('b',[1],initializer=tf.constant_initializer(0.0))

# [None,3072] * [3072,1] = [None,1]
# only y_ f(x) = x * W + b [None,1]
y_ = tf.matmul(x,w) + b

# [None,1]
# change f(x) output in (0,1) range
p_y_1 = tf.nn.sigmoid(y_)

# [None,1]
y_reshaped = tf.reshape(y,(-1,1))
y_reshaped_float = tf.cast(y_reshaped,tf.float32)

# avg(y_prep**2 - y_expect**2) loss = L(f(x) - y)
# cal loss
loss = tf.reduce_mean(tf.square(y_reshaped_float - p_y_1))

# bool 
# predict return bool value so need bool type as int64
predict = p_y_1 > 0.5
# equal 
correct_prediction = tf.equal(tf.cast(predict,tf.int64),y_reshaped)

# [1,0,0,1,1,]
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float64))

with tf.name_scope('train_op'):
    # learning rate le-3 
    train_op = tf.train.AdamOptimizer(le-3).minimize(loss)
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run([loss,accuracy,train_op],feed_dict={x:,y: })
    