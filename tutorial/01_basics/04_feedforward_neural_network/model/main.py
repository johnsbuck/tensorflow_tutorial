#!usr/bin/env python2

from __future__ import print_function
from model import FNN
from tensorflow.examples.tutorials.mnist import input_data

# ------------------------------------------------
print("==> Importing MNIST")

mnist = input_data.read_data_sets('/tmp/tensorflow/mnist/input_data', one_hot=True)

# ------------------------------------------------
print("==> Creating Feedforward Neural Network (FNN) Model")

print("-> Creating placeholders for input & output")
print("# Inputs   : 784   (28x28)")
print("# Outputs  : 10    (0-9)")

print("-> Defining model structure")
fnn = FNN(784, 10, [500])
# ------------------------------------------------
print("==> Training FNN Model")

X, y = mnist.train.next_batch(10000)
fnn.train(X, y, n_epochs=200, batch_size=50, epoch_print=10)

# ------------------------------------------------
print("==> Testing FNN Model")
print(fnn.accuracy(mnist.test.images, mnist.test.labels))
