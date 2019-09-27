import numpy as np
from matplotlib import pyplot as plt

X = np.random.uniform(50,100,100)
# Y = np.dot(X,2) + 0.5 + np.random.random(-5,5)
Y = np.dot(X,2) + 5  + np.random.randint(-50,50)
print(X[0],np.dot(X,2)[0],Y[0])
plt.scatter(X,np.dot(X,2) + 5  + np.random.randint(-50,50))

plt.show()