import tensorflow as tf
# print(tf.__version__)
import numpy as np
import pickle
import os
# save file format as numpy
# file format as cPickle
CIFAR_DIR = "./cifar-10-batches-py"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# print(os.listdir(CIFAR_DIR))

# ['data_batch_1', 'readme.html', 'batches.meta', 'data_batch_2', 'data_batch_5', 'test_batch', 'data_batch_4', 'data_batch_3']

with open(os.path.join(CIFAR_DIR,"data_batch_1"),'rb') as f:
    data = pickle.load(f)
    print(type(data))
    print(data.keys())
    print(type(data['data']))
    print(type(data['labels']))
    print(type(data['batch_label']))
    print(type(data['filenames']))
    print(data['data'].shape)
    print(data['data'][0:2])
    print(data['labels'][0:2])
    print(data['batch_label'])
    print(data['filenames'][0:2])

image_arr = data['data'][100]
image_arr = image_arr.reshape((3,32,32))
image_arr = image_arr.transpose((1,2,0))

import matplotlib.pyplot as plt
# plt.imshow(image_arr)
# plt.show()