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
class CifarData:
    def __init__(self,filenames,need_shuffle):
        all_data = []
        all_labels = []
        for filename in filenames:
            data,labels = load_data(filename)
            all_data.append(data)
            all_labels.append(labels)
        self._data = np.vstack(all_data)
        self._data = self._data/127.5 - 1
        self._labels = np.hstack(all_labels)
        # print self._data.shape
        # print self._labels.shape
        self._num_examples = self._data.shape[0]
        self._need_shuffle = need_shuffle
        self._indicator = 0
        if self._need_shuffle:
            self._shuffle_data()
    
    def _shuffle_data(self):
        p = np.random.permutation(self._num_examples)
        self._data = self._data[p]
        self._labels = self._labels[p]
    
    def next_batch(self,batch_size):
        """ return batch_size examples as a batch. """
        end_indicator = self._indicator + batch_size
        if end_indicator > self._num_examples:
            if self._need_shuffle:
                self._shuffle_data()
                self._indicator = 0
                end_indicator = batch_size
            else:
                raise Exception("have no more examples")
        if end_indicator > self._num_examples:
            raise Exception("batch size is larager than all examples")
        batch_data = self._data[self._indicator:end_indicator]
        batch_labels = self._labels[self._indicator:end_indicator]
        self._indicator = end_indicator
        return batch_data, batch_labels

train_filenames = [os.path.join(CIFAR_DIR,'data_batch_%d' % i) for i in range(1,6)]
test_filenames = [os.path.join(CIFAR_DIR,'test_batch')]


# batch_data, batch_labels = train_data.next_batch(10)
# print batch_data
# print batch_labels
# (10000, 3072)
# (10000,)

# test_data = CifarData(train_filenames,True)

