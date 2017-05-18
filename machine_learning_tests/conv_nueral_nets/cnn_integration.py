import cPickle as pickle
import gzip
import os
from urllib import urlretrieve

import archaea.machine_learning_tests.test_data.cnn_test_data as constants
import matplotlib.cm as cm
import numpy as np
from numpy import array, shape

import archaea.machine_learning.conv_nn.network_builder as builder


def load_dataset():
    url = 'http://deeplearning.net/data/mnist/mnist.pkl.gz'
    filename = 'mnist.pkl.gz'
    if not os.path.exists(filename):
        print("Downloading MNIST dataset...")
        urlretrieve(url, filename)
    with gzip.open(filename, 'rb') as f:
        data = pickle.load(f)
    X_train, y_train = data[0]
    print X_train
    print y_train
    X_val, y_val = data[1]
    X_test, y_test = data[2]
    X_train = X_train.reshape((-1, 1, 28, 28))
    print X_train
    X_val = X_val.reshape((-1, 1, 28, 28))
    X_test = X_test.reshape((-1, 1, 28, 28))
    y_train = y_train.astype(np.uint8)
    y_val = y_val.astype(np.uint8)
    y_test = y_test.astype(np.uint8)
    return X_train, y_train, X_val, y_val, X_test, y_test



X_train, y_train, X_val, y_val, X_test, y_test = load_dataset()

#plt.imshow(X_train[0][0], cmap=cm.binary)

#net1 = builder.ConvNetworkBuilder(constants.CONV_NN_PARAMETERS).build()

# Train the network
#nn = net1.fit(X_train, y_train)
#print X_test

#preds = net1.predict(X_test)

"""

visualize.plot_conv_weights(net1.layers_['conv2d1'])

dense_layer = layers.get_output(net1.layers_['dense'], deterministic=True)
output_layer = layers.get_output(net1.layers_['output'], deterministic=True)
input_var = net1.layers_['input'].input_var
f_output = theano.function([input_var], output_layer)
f_dense = theano.function([input_var], dense_layer)

instance = X_test[0][None, :, :]
pred = f_output(instance)
N = pred.shape[1]
plt.bar(range(N), pred.ravel())
"""
