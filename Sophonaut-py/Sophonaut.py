OutputDir = "C:\Users\Jordan\Desktop\output"
TrainingDataDir = "C:\Users\Jordan\Desktop\cifar-10-batches-py"
LabelFile = "batches.meta"
FileName = "data_batch_1"

Width, Height = 32, 32
Rows = 10000
RowSize = 3073

import numpy as np
import matplotlib.pyplot as pl

from os.path import join
import cPickle

from util import *

def load_batch(file_name):
    with open(file_name, 'rb') as file:
        return cPickle.load(file)

def Main():
    # Read label meta file.
    _ = load_batch(join(TrainingDataDir, LabelFile))
    LabelNames = _['label_names']

    assert len(LabelNames) == 10

    # Read training data.
    _ = load_batch(join(TrainingDataDir, FileName))
    images, labels = _['data'], _['labels']

    assert images.shape == (Rows, 32 * 32 * 3)

    # Reshape data.
    images = np.reshape(images, (Rows, 3, 32, 32)) / 255.0
    images = images.swapaxes(1, 3).swapaxes(1, 2)

    # Unpack blocks.
    blocks = rolling_window(images, (1, 2, 2, 3))
    blocks = np.reshape(blocks, (Rows * (Width - 1) * (Height - 1), 12))
    blocks = blocks * 1.0
    #blocks = list_normalized(blocks)

    find_kernel(blocks)
    print "Done!"

def test_kernel(blocks, kernel, threshold):
    w, b = kernel

    #similiarty = list_similarity(blocks - b, w)
    #num_well_fit = (similiarty > threshold).sum()
    #ratio = num_well_fit / float(blocks.shape[0])
    #print 'Kernel result: {pct}% above {threshold} threshold.'.format(pct=100*ratio,threshold=threshold)

    error = list_error(blocks - b, w)
    num_well_fit = (error < threshold).sum()

    ratio = num_well_fit / float(blocks.shape[0])

    print 'Kernel result: {pct}% below {threshold} error threshold.'.format(pct=100*ratio,threshold=threshold)

    return ratio

def find_kernel(blocks):
    w = np.ones(12)
    w = normalized(w)
    b = np.ones(12) * .5

    #w = blocks[47291]
    #w = normalized(w-b)

    test_kernel(blocks, (w, b), .5)
    test_kernel(blocks, (w, b), .3)

    return (w, b)

Main()