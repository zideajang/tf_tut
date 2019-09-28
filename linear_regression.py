import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D

learning_rate = 0.01
epochs = 200

n_samples = 30
train_x = np.linspace(0,20,n_samples)
train_y = 3 * train_x + 4* np.random.randn(n_samples)

# plt.plot(train_x,train_y,'o')
# plt.plot(train_x, 3 * train_x)
# plt.show()
# fig = plt.figure()
# ax1 = plt.axes(projection='3d')

# fig=plt.figure()

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

W = tf.Variable(np.random.randn,name='weights')
B = tf.Variable(np.random.randn,name='bias')

# pred = tf.add(tf.multiply(X,W),B)
pred = X * W + B

cost = tf.reduce_sum((pred - Y) ** 2) / (2 * n_samples)

optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

init = tf.global_variables_initializer()

W1 = np.empty(10)
C1 = np.empty(10)
index = 0
with tf.Session() as sess:
    sess.run(init)
    for epoch in range(epochs):
        for x, y in zip(train_x,train_y):
            sess.run(optimizer,feed_dict={X:x,Y:y})
        if not epoch % 20:
            c = sess.run(cost,feed_dict={X:train_x,Y:train_y})
            w = sess.run(W)
            b = sess.run(B)
            # print(c)
            W1[index] = w
            C1[index] = c
            index = index + 1
            print(f'epoch:{epoch:04d} c={c:.4f} w={w:.4f} b={b:.4f}')
            
            # ax1.scatter3D(w,b,c, cmap='Blues')  
    plt.plot(W1,C1)
    plt.show()

    # weight = sess.run(W)
    # bias = sess.run(B)
    # plt.plot(train_x,train_y,'o')
    # plt.plot(train_x,weight * train_x + bias)
    # plt.show()

