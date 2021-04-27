import scipy.io
import os
import numpy as np


def minmax_normalize(array):    
    amin = np.min(array)
    amax = np.max(array)
    return (array - amin) / (amax - amin)


if __name__ == '__main__':
    data = scipy.io.loadmat('./data/dc/dc.mat')['img']
    data = data.astype(np.float32)
    h, w, c = data.shape
    for i in range(c):
        data[:, :, i] = minmax_normalize(data[:, :, i])
    scipy.io.savemat('./data/dc/dc_norm.mat', {'img': data})

