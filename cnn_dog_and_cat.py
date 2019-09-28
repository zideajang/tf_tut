import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import pickle

import random

DATADIR = "PetImages"
CATEGORIES = ['Dog','Cat']
IMG_SIZE = 50
training_data = []
def create_training_data():
    for category in CATEGORIES:
        # print(category)
        path = os.path.join(DATADIR,category) # path to cats or dogs dir
        # print(path)
        class_num = CATEGORIES.index(category)
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path,img),cv2.IMREAD_GRAYSCALE)
                new_array = cv2.resize(img_array,(IMG_SIZE,IMG_SIZE))
                training_data.append([new_array,class_num])
            except Exception as e:
                pass
            # plt.imshow(img_array,cmap="gray")
            # plt.show()
create_training_data()
print(len(training_data))

random.shuffle(training_data)

for sample in training_data[:10]:
    print(sample[1])

X = [] # Set
y = [] #label
for features, label in training_data:
    X.append(features)
    y.append(label)
X = np.array(X).reshape(-1,IMG_SIZE,IMG_SIZE,1)
# IMG_SIZE = 50
# print(img_array.shape)
# new_array = cv2.resize(img_array,(IMG_SIZE,IMG_SIZE))
# plt.imshow(new_array,cmap='gray')
# plt.show()

pickle_out = open("X.pickle","wb")
pickle.dump(X, pickle_out)
pickle_out.close()

pickle_out = open("y.pickle","wb")
pickle.dump(y, pickle_out)
pickle_out.close()

pickle_in = open("X.pickle","rb")
X = pickle.load(pickle_in)
print(X[1])


