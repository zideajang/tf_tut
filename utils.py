import numpy as np
import cPickle
import os
import matplotlib.pyplot as plt
# save file format as numpy
# file format as cPickle
CIFAR_DIR = "./cifar-10-batches"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

print os.listdir(CIFAR_DIR)

def load_data(filename):
    """ read data from data file. """
    with open(filename,'rb') as f:
        data = cPickle.load(f)
        return data['data'],data['labels']
# with open(os.path.join(CIFAR_DIR,"data_batch_1"),'rb') as f:
#     data = pickle.load(f)
# image_arr = data['data'][100]
# image_arr = image_arr.reshape((3,32,32))
# image_arr = image_arr.transpose((1,2,0))
# plt.imshow(image_arr)
# plt.show()