import numpy as np
import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
a = np.array([0.1,0.6,0.8])
print a
b = a > 0.5
print b
c = tf.cast(b,tf.int64)
# print 
pret = np.array([0,1,0])
result = tf.equal(c,pret)

with tf.Session() as sess:
    result = sess.run(result)
    print result