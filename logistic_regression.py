import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

(x_train,y_train),(x_test,y_test) = tf.keras.datasets.mnist.load_data()

# fig,axes = plt.subplots(1,4,figsize=(7,3))
# for img, label, ax in zip(x_train[:4],y_train[:4],axes):
#     ax.set_title(label)
#     ax.imshow(img)
#     ax.axis('off')
# plt.show()

# print(f'train images: {x_train.shape}')
# print(f'train labels: {y_train.shape}')
# print(f'test images: {x_test.shape}')
# print(f'test labels: {x_test.shape}')

x_train = x_train.reshape(60000,784) / 255
x_test = x_test.reshape(10000,784) / 255

with tf.Session() as sess:
    y_train = sess.run(tf.one_hot(y_train,10))
    y_test = sess.run(tf.one_hot(y_test,10))

# print(y_train[:4])

learning_rate = 0.01
epochs = 50
batch_size = 100
batches = int(x_train.shape[0] / batch_size)

# inputs
# X
X = tf.placeholder(tf.float32,[None,784])
Y = tf.placeholder(tf.float32,[None,10])

W = tf.Variable(np.random.randn(784,10).astype(np.float32))
B = tf.Variable(np.random.randn(10).astype(np.float32))

# prediction
pred = tf.nn.softmax(tf.add(tf.matmul(X,W),B))

# Loss
cost = tf.reduce_mean(-tf.reduce_sum(Y* tf.log(pred),axis=1))
# 

# x = np.linspace(1/100,1,100)
# plt.plot(x,np.log(x))
# plt.show()


# pred
a = np.log([[0.04,0.13,0.96,0.12], #correct
        [0.01,0.93,0.06,0.07]]) #incrroet
# labels
b = np.array([[0,0,1,0],
        [1,0,0,0]])        

# print(-a * b)
"""
r_sum = np.sum(-a * b, axis=1)
r_mean = np.mean(r_sum)
print(r_sum)
print(r_mean)
"""

"""
[0.04082199 4.60517019]
2.322996090254173
"""

with tf.Session() as sess:
    tf_sum = sess.run(-tf.reduce_mean(a * b,axis=1))
    tf_mean = sess.run(tf.reduce_mean(tf_sum))

print(f'sum = {tf_sume}')
print(f'mean = {tf_mean}')
