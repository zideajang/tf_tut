import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# create train data
X1 = np.random.uniform(50,100,100)
X2 = np.random.uniform(100,200,100)

fig = plt.figure()
ax1 = plt.axes(projection='3d')

fig=plt.figure()

Y = np.dot(X1,2) + np.dot(X2,3) + 5 + np.random.randint(-5,5)

print(Y)

ax1.scatter3D(X1,X2,Y, cmap='Blues')  
plt.show()
