import numpy as np
import pickle
import os
# save file format as numpy
# file format as cPickle
CIFAR_DIR = "./cifar-10-batches-py"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def load_data(filename):
    """read data from cifar file"""
    with open(os.path.join(CIFAR_DIR,"data_batch_1"),'rb') as f:
        data = pickle.load(f)
        return data['data'],data['labels']

class CifarData:
    # need_shuffle control is shuffle for train set and test set
    def __init__(self, filenames, need_shuffle):
        all_data = []
        all_labels = []
        # iterate filename get data and labels
        for filename in filenames:
            data, labels = load_data(filename)
            all_data.append(data)
            all_labels.append(all_labels)
        self._data = np.vstack(all_data)
        self._data = self._data / 127.5 - 1
        self._labels = np.hstack(all_labels)
        # print(self._data.shape)
        # print(self._labels.shape)
        self._num_examples = self._data.shape[0]
        self._need_shuffle = need_shuffle
        self._indicator = 0
        if self._need_shuffle:
            self._shuffle_data()

    def _shuffle_data(self):
        p = np.random.permutation(self._num_examples)
        self._data = self._data[p]
        self._labels = self._labels[p]

    def next_batch(self, batch_size):
        end_indicator = self._indicator + batch_size
        if end_indicator > self._num_examples:
            if self._need_shuffle:
                self._shuffle_data()
                self._indicator = 0
                end_indicator = batch_size
            else:
                raise Exception("have no more examples")
        if end_indicator > self._num_examples:
            raise Exception("batch size is larger than all examples")
        batch_data = self._data[self._indicator:end_indicator]
        batch_labels = self._labels[self._indicator:end_indicator]
        self._indicator = end_indicator
        return batch_data, batch_labels
        
