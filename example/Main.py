import numpy as np
import pickle

from backend.dense import Dense
from backend.activation import Sigmoid
from backend.train import train
from backend.test import test

def read():
    import struct
    def read_idx(filename):
        with open(filename, 'rb') as f:
            zero, data_type, dims = struct.unpack('>HBB', f.read(4))
            shape = tuple(struct.unpack('>I', f.read(4))[0] for d in range(dims))
            return np.frombuffer(f.read(), dtype=np.uint8).reshape(shape)
    train_images_path = 'data/train-images.idx3-ubyte'
    train_labels_path = 'data/train-labels.idx1-ubyte'
    test_images_path = 'data/t10k-images.idx3-ubyte'
    test_labels_path = 'data/t10k-labels.idx1-ubyte'
    train_images = read_idx(train_images_path)
    train_labels = read_idx(train_labels_path)
    test_images = read_idx(test_images_path)
    test_labels = read_idx(test_labels_path)
    return (train_images, train_labels, test_images, test_labels)

DATA = read()

train_data = DATA[0] / 255
train_data = train_data.reshape(len(train_data), 784, 1)

test_data = DATA[2] / 255
test_data = test_data.reshape(len(test_data), 784, 1)

train_label = DATA[1]
tmp = np.zeros((len(train_label), 10, 1))
for i in range(len(train_label)):
    tmp[i][train_label[i]] = 1
train_label = tmp

test_label = DATA[3]
tmp = np.zeros((len(test_label), 10, 1))
for i in range(len(test_label)):
    tmp[i][test_label[i]] = 1
test_label = tmp

file = open("parameters.dat", "rb")
network = [
    Dense(784, 392, pickle.load(file), pickle.load(file)),
    Sigmoid(),
    Dense(392, 50, pickle.load(file), pickle.load(file)),
    Sigmoid(),
    Dense(50, 10, pickle.load(file), pickle.load(file)),
    Sigmoid()
]
file.close()

train(network, train_data, train_label, 1)
test(network, test_data, test_label)
