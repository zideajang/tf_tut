import tensorflow as tf
from numpy.random import RandomState

batch_size = 8

# [2,3] [[[0.1,0.2]]
w1 = tf.Variable(tf.random_normal([2,3],stddev=1,seed=1))
w2 = tf.Variable(tf.random_normal([3,1],stddev=1,seed=1))

# [None,2] [[1,2],[2,3]]
x = tf.placeholder(tf.float32,shape=(None,2),name='x-input')
y_ = tf.placeholder(tf.float32,shape=(None,1),name='y-input')

a = tf.matmul(x,w1)
y = tf.matmul(a,w2)

cross_entropy = -tf.reduce_mean(
        y_ * tf.log(tf.clip_by_value(y,1e-10,1.0)))
train_step = tf.train.AdagradOptimizer(0.001).minimize(cross_entropy)

rdm = RandomState(1)
dataset_size = 128
X = rdm.rand(dataset_size,2)
Y = [[int(x1 + x2 < 1)] for (x1,x2) in X ]

init_op = tf.initialize_all_variables()
with tf.Session() as sess:
    sess.run(init_op)
    print(sess(w1))
    print(sess(w2))
    