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

# None present confirm count of smaples

# reduce size of smapling 
def residual_block(x,output_channel):
    """ residual connection implementation """
    # 
    input_channel = x.get_shape().as_list()[-1]
    if input_channel * 2 == output_channel:
        increase_dim = True
        strides = (2,2)
    elif input_channel == output_channel:
        increase_dim = False
        strides = (1,1)
    else:
        raise Exception("input channel can't match output channel")
    conv1 = tf.layers.conv2d(x,
                output_channel,
                (3,3),
                strides = strides,
                padding= 'same',
                activation = tf.nn.relu,
                name='conv1')
    conv2 = tf.layers.conv2d(conv1,
                output_channel,
                (3,3),
                strides = (1,1),
                padding= 'same',
                activation = tf.nn.relu,
                name='conv2')
    if increase_dim:
        # [None,image_width,image_height,channel]
        pooled_x = tf.layers.average_pooling2d(x,(2,2),(2,2),padding='valid')
        padded_x = tf.pad(pooled_x,
                        [[0,0],[0,0],[0,0],[input_channel // 2,input_channel // 2]])
    else:
        padded_x = x
    output_x = conv2 + padded_x
    return output_x
def res_net(x,
            num_residual_blocks,
            num_filter_base,
            class_num):
    """ residual network implementation """
    """
        x
        num_residual_blocks: eg:[3,4,6,3] define block
        num_filter_base:
        class_num:
    """
    num_subsampling = len(num_residual_blocks)
    layers = []
    # X: [None, width, height, channel ] -> width height channel
    input_size = x.get_shape().as_list()[1:]
    with tf.variable_scope('conv0'):
        conv0 = tf.layers.conv2d(x,num_filter_base,(3,3),strides=(1,1),padding='same',activation=tf.nn.relu,name='conv0')

        layers.append(conv0)

    # num_subsampling = 4 sample_id = [0,1,2,3]
    for sample_id in range(num_subsampling):
        for i in range(num_residual_blocks[sample_id]):
            with tf.variable_scope('conv%d_%d' % (sample_id,i)):
                conv = residual_block(layers[-1],num_filter_base*(2 ** sample_id))
                layers.append(conv)
    multiplier =  2 ** (num_subsampling - 1 )
    assert layers[-1].get_shape().as_list()[1:] == [input_size[0] / multiplier, input_size[1] / multiplier, num_filter_base * multiplier]

    with tf.variable_scope('fc'):
        # layer[-1].shape : [None, width, height, channel]
        global_pool = tf.reduce_mean(layers[-1],[1,2])
        logits = tf.layers.dense(global_pool,class_num)
        layers.append(logits)
    return layers[-1]
x = tf.placeholder(tf.float32,[None,3072])
y = tf.placeholder(tf.int64,[None])
x_image = tf.reshape(x,[-1,3,32,32])
# 32 * 32
x_image = tf.transpose(x_image,perm=[0,2,3,1])

# y_ = tf.layers.dense(flatten,10)
y_ = res_net(x_image, [2,3,2],32,10)
"""
hidden1 = tf.layers.dense(x,100,activation=tf.nn.relu)
hidden2 = tf.layers.dense(hidden1,100,activation=tf.nn.relu)
hidden3 = tf.layers.dense(hidden2,50,activation=tf.nn.relu)
[Train] Step: 9999, loss: 1.24794, acc: 0.45000
[Test ] Step: 10000, acc: 0.50300
"""
# simplify with tensorflow api
# y_ = tf.layers.dense(hidden3,10)

loss = tf.losses.sparse_softmax_cross_entropy(labels=y,logits=y_)

# indices
predict = tf.argmax(y_,1)
# equal 
correct_prediction = tf.equal(predict,y)

# [1,0,0,1,1,]
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float64))

train_data = CifarData(train_filenames,True)
# test_data = CifarData(test_filenames,False)
batch_size = 20
train_steps = 1000
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