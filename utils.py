import numpy as np
from RAM_parameters import *

def dense_to_one_hot(labels, num_classes=10):
    labels_one_hot = np.zeros((batch_size, num_classes))
    for i in range(batch_size):
        labels_one_hot[i][labels[i]] = 1
    return labels_one_hot
