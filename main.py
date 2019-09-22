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
# (3071,10)
w = tf.get_variable('w',[x.get_shape()[-1],10],initializer=tf.random_normal_initializer(0,1))
# (10,)
b = tf.get_variable('b',[10],initializer=tf.constant_initializer(0.0))

# [None,3072] * [3072,10] = [None,10]
# only y_ f(x) = x * W + b [None,1]
y_ = tf.matmul(x,w) + b
"""
# [None,1]
# change f(x) output in (0,1) range
p_y_1 = tf.nn.sigmoid(y_)
# p_y_1 = tf.nn.sigmoid(y_)

# [None,1]
y_reshaped = tf.reshape(y,(-1,1))
y_reshaped_float = tf.cast(y_reshaped,tf.float32)

# avg(y_prep**2 - y_expect**2) loss = L(f(x) - y)
# cal loss
loss = tf.reduce_mean(tf.square(y_reshaped_float - p_y_1))

"""
# 1 + e^x
# e^x / sum(e^x)
# [[0.01,0.02...0.09],[],...]
p_y = tf.nn.softmax(y_)
# 5 -> [0,0,0,0,1,..,0]
y_one_hot = tf.one_hot(y,10,dtype=tf.float32)
loss = tf.reduce_mean(tf.square(y_one_hot - p_y))

# bool 
# predict return bool value so need bool type as int64
# predict = p_y_1 > 0.5

# indices
predict = tf.argmax(y_,1)
# equal 
correct_prediction = tf.equal(predict,y)

# [1,0,0,1,1,]
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float64))

train_data = CifarData(train_filenames,True)
# test_data = CifarData(test_filenames,False)
batch_size = 20
train_steps = 10000
test_steps = 100
with tf.name_scope('train_op'):
    # learning rate le-3 
    train_op = tf.train.AdamOptimizer(1e-3).minimize(loss)
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for i in range(train_steps):
        batch_data,batch_labels = train_data.next_batch(batch_size)
        loss_val, accu_val,_ = sess.run([loss,accuracy,train_op],feed_dict={x: batch_data,y: batch_labels})
        if (i+1) % 500 == 0:
            print '[Train] Step: %d, loss: %4.5f, acc: %4.5f' % (i, loss_val, accu_val)
        if (i+1) % 1000 == 0:
            test_data  = CifarData(test_filenames,False)
            all_test_acc_val = []
            for j in range(test_steps):
                test_batch_data,test_batch_labels = test_data.next_batch(batch_size)
                test_acc_val = sess.run([accuracy],
                    feed_dict={x: test_batch_data,y: test_batch_labels })
                all_test_acc_val.append(test_acc_val)
            test_acc = np.mean(all_test_acc_val)
            print '[Test ] Step: %d, acc: %4.5f ' % (i+1,test_acc)