import tensorflow as tf
# print(tf.__version__)
import numpy as np
import cPickle
import os
import utils
from utils_aug import CifarData
# save file format as numpy
# file format as cPickle
CIFAR_DIR = "./cifar-10-batches"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# print(os.listdir(CIFAR_DIR))

# tensorboard

train_filenames = [os.path.join(CIFAR_DIR, 'data_batch_%d' % i)
                   for i in range(1, 6)]
test_filenames = [os.path.join(CIFAR_DIR, 'test_batch')]
train_data = CifarData(train_filenames, True)
test_data = CifarData(test_filenames, False)

batch_size = 20

# None present confirm count of smaples
x = tf.placeholder(tf.float32,[batch_size,3072])
y = tf.placeholder(tf.int64,[batch_size])

x_image = tf.reshape(x,[-1,3,32,32])
# 32 * 32
x_image = tf.transpose(x_image,perm=[0,2,3,1])

x_image_arr = tf.split(x_image,num_or_size_splits=batch_size,axis=0)
result_x_image_arr = []
for x_single_image in x_image_arr:
    # single image [1,32,32,3]
    x_single_image = tf.reshape(x_single_image,[32,32,3])
    data_aug_1 = tf.image.random_flip_left_right(x_single_image)
    data_aug_2 = tf.image.random_brightness(data_aug_1,max_delta=63)
    data_aug_3 = tf.image.random_contrast(data_aug_2,lower=0.2,upper=1.8)
    x_single_image = tf.reshape(data_aug_3,[1,32,32,3])
    result_x_image_arr.append(x_single_image)

result_x_images = tf.concat(result_x_image_arr,axis=0)
normal_result_x_images = result_x_images / 127.5 - 1

# image argument

def conv_wrapper(inputs,name,output_channel=32, kernel_size=(3,3)):
    
    return tf.layers.conv2d(
            inputs,
            output_channel,
            kernel_size,
            padding='same',
            activation=tf.nn.relu,
            name=name)

# neru feature_map image output
conv1_1 = tf.layers.conv2d(normal_result_x_images,32,(3,3),padding='same',activation=tf.nn.relu,name='conv1_1')
conv1_2 = tf.layers.conv2d(conv1_1,32,(3,3),padding='same',activation=tf.nn.relu,name='conv1_2')
conv1_3 = tf.layers.conv2d(conv1_2,32,(3,3),padding='same',activation=tf.nn.relu,name='conv1_3')
# 16 * 16
pooling1 = tf.layers.max_pooling2d(conv1_3,(2,2),(2,2),name='pool1')

conv2_1 = tf.layers.conv2d(pooling1,32,(3,3),padding='same',activation=tf.nn.relu,name='conv2_1')
conv2_2 = tf.layers.conv2d(conv2_1,32,(3,3),padding='same',activation=tf.nn.relu,name='conv2_2')
conv2_3 = tf.layers.conv2d(conv2_2,32,(3,3),padding='same',activation=tf.nn.relu,name='conv2_3')
# 8 * 8
pooling2 = tf.layers.max_pooling2d(conv2_3,(2,2),(2,2),name='pool2')

conv3_1 = tf.layers.conv2d(pooling2,32,(3,3),padding='same',activation=tf.nn.relu,name='conv3_1')
conv3_2 = tf.layers.conv2d(conv3_1,32,(3,3),padding='same',activation=tf.nn.relu,name='conv3_2')
conv3_3 = tf.layers.conv2d(conv3_2,32,(3,3),padding='same',activation=tf.nn.relu,name='conv3_3')
# 4 * 4
pooling3 = tf.layers.max_pooling2d(conv3_3,(2,2),(2,2),name='pool3')

# [None , 4 * 4 * 32]
flatten = tf.layers.flatten(pooling3)
y_ = tf.layers.dense(flatten,10)
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
# batch_size = 20
train_steps = 10000
test_steps = 100

with tf.name_scope('train_op'):
    # learning rate le-3 
    train_op = tf.train.AdamOptimizer(1e-3).minimize(loss)
init = tf.global_variables_initializer()


def variable_summary(var, name):
    """ constructs summary for statistics of a variable """
    with tf.name_scope(name):
        mean = tf.reduce_mean(var)
        with tf.name_scope("stddev"):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('mean',mean)
        tf.summary.scalar('stddev',stddev)
        tf.summary.scalar('min',tf.reduce_mean(var))
        tf.summary.scalar('max',tf.reduce_max(var))
        tf.summary.histogram('histogram',var)
with tf.name_scope('summary'):
    variable_summary(conv1_1,'conv1_1')
    variable_summary(conv1_2,'conv1_2')
    variable_summary(conv2_1,'conv2_1')
    variable_summary(conv2_2,'conv2_2')
    variable_summary(conv3_1,'conv3_1')
    variable_summary(conv3_2,'conv3_2')

# visilization loss and accuracy
loss_summary = tf.summary.scalar('loss',loss)
# 'loss' <10,1.1> <20 1.08
accuracy_summary = tf.summary.scalar('accuracy',accuracy)
# inverse regerization 
# source_image = (x_image + 1) * 127.5
inputs_summary = tf.summary.image('inputs_image',result_x_images)

# call summary to merget
merged_summary = tf.summary.merge_all()

merged_summary_test = tf.summary.merge([loss_summary,accuracy_summary])

LOG_DIR = '.'
run_label = 'run_vgg_tensorboard'
run_dir = os.path.join(LOG_DIR,run_label)
if not os.path.exists(run_dir):
    os.mkdir(run_dir)
train_log_dir = os.path.join(run_dir,'train')
test_log_dir = os.path.join(run_dir,'test')

if not os.path.exists(train_log_dir):
    os.mkdir(train_log_dir)
if not os.path.exists(test_log_dir):
    os.mkdir(test_log_dir)

output_summary_every_steps = 100    

with tf.Session() as sess:
    sess.run(init)
    
    train_writer = tf.summary.FileWriter(train_log_dir,sess.graph)
    test_writer = tf.summary.FileWriter(test_log_dir)

    fixed_test_batach_data,fixed_test_batch_labels = test_data.next_batch(batch_size)

    # test_writer = tf.sum
    for i in range(train_steps):
        batch_data,batch_labels = train_data.next_batch(batch_size)
        eval_ops = [loss,accuracy,train_op]
        should_output_summary = ((i+1) % output_summary_every_steps == 0 )
        if should_output_summary:
            eval_ops.append(merged_summary)
        eval_ops_results = sess.run(eval_ops,feed_dict={x: batch_data,y: batch_labels})
        # 
        loss_val, accu_val = eval_ops_results[0:2]
        
        if should_output_summary:
            train_summary_str = eval_ops_results[-1]
            train_writer.add_summary(train_summary_str,i+1)
            test_summary_str = sess.run([merged_summary_test],feed_dict={x:fixed_test_batach_data, y:fixed_test_batch_labels})[0]
            test_writer.add_summary(test_summary_str,i+1)

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