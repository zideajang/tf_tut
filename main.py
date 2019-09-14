import tensorflow as tf
# print(tf.__version__)
import numpy as np
import cPickle
import os
# save file format as numpy
# file format as cPickle
CIFAR_DIR = "./cifar-10-batches-py"

print(os.listdir(CIFAR_DIR))

# ['data_batch_1', 'readme.html', 'batches.meta', 'data_batch_2', 'data_batch_5', 'test_batch', 'data_batch_4', 'data_batch_3']