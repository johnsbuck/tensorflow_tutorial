#!usr/bin/env python2

"""Convolutional Neural Network
This is an example of a convolutional neural network on TensorFlow using MNIST dataset.
"""

from __future__ import print_function
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import tempfile


def weight_variables(shape):
    """ Used to generate variables used for weights in the CNN.

    Args:
        shape: List(Integer). Used to generate a normal tensor.

    Returns:
        Variable
    """
    initial = tf.random_normal(shape, stddev=tf.sqrt(2./shape[1]))
    return tf.Variable(initial, name="W")


def bias_variables(shape, constant=0.01):
    """ Used to generate variables used for biases in the CNN.

    Args:
        shape: List(Integer). Used to generate a constant tensor.
        constant: A float or integer used to define all values in the returned variable.

    Returns:
        Variable
    """
    initial = tf.constant(constant, shape=shape)
    return tf.Variable(initial, name="B")

def conv(input, w_shape):
    """Used to implement a Convolutional Layer.

    Args:
        input: Previous tensor or placeholder used by model
        w_shape: List. Defines the weight and bias variables

    Returns:
        Tensor
    """
    W = weight_variables(w_shape)
    B = bias_variables([w_shape[len(w_shape) - 1]])
    tf.summary.image("Filters", W[:, :, :, :1], max_outputs=1)
    tf.summary.histogram("W", W)
    tf.summary.histogram("B", B)
    h = tf.nn.relu(tf.nn.conv2d(input, W, strides=[1, 1, 1, 1], padding="SAME") + B)
    return h


def pool(input):
    """Used to implement a Max-Pooling Layer.
    Down-samples a feature in half using max-pooling.

        Args:
            x: Previous tensor or placeholder used by model

        Returns:
            Tensor
        """
    return tf.nn.max_pool(input, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")


def linear(input, in_size, out_size):
    """Used to implement a Linear layer

    Args:
        input: Previous tensor or placeholder used by model
        in_size: Integer. Used to define size of previous output
        out_size: Integer. Used to define size of current output

    Returns:
        Tensor
    """
    W = weight_variables([in_size, out_size])
    B = bias_variables([out_size])
    tf.summary.histogram("W", W)
    tf.summary.histogram("B", B)
    return tf.matmul(input, W) + B


def cnn():
    """Defines a convolutional neural network and placeholders.

    Returns:
        Tuple. (model, x, y, keep_prob)
            model: Tensor. Convolutional Neural Network.
            x: Placeholder. Takes in a Tensor of shape (Batch_Size, 784).
            y: Placeholder. Takes in a Tensor of shape (Batch_Size, 10). Used for loss function.
            keep_prob: Placeholder. Takes in a tf.float32 to define the probability of the dropout layer.
    """
    x = tf.placeholder(tf.float32, [None, 784])
    y = tf.placeholder(tf.float32, [None, 10])

    with tf.name_scope("reshape"):
        x_image = tf.reshape(x, [-1, 28, 28, 1])    # Reshapes 1D input vector to 2D batch of images

    with tf.name_scope("conv1"):  # Used for graphing in tensorboard
        h_conv1 = conv(x_image, [5, 5, 1, 32])

    with tf.name_scope("pool1"):
        h_pool1 = pool(h_conv1)

    with tf.name_scope("conv2"):
        h_conv2 = conv(h_pool1, [5, 5, 32, 64])

    with tf.name_scope("pool2"):
        h_pool2 = pool(h_conv2)

    with tf.name_scope("fc1"):
        h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])  # Flattens for fully connected layer
        h_fc1 = tf.nn.relu(linear(h_pool2_flat, 7 * 7 * 64, 1024))
        tf.summary.histogram("activation", h_fc1)

    with tf.name_scope("dropout"):
        keep_prob = tf.placeholder(tf.float32)  # Used to change dropout from train (0.5) and evaluate (1.0)
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
        tf.summary.histogram("dropout", h_fc1_drop)

    with tf.name_scope("fc2"):
        h_fc2 = linear(h_fc1_drop, 1024, 10)
        tf.summary.histogram("activation", h_fc2)

    return h_fc2, x, y, keep_prob

# --------------------------------------------------------------
print("==> Importing MNIST")

mnist = input_data.read_data_sets('/tmp/tensorflow/mnist/input_data', one_hot=True)

# --------------------------------------------------------------
print("==> Creating Feedforward Neural Network (FNN) Model")

print("-> Creating placeholders for input & output")
print("# Inputs   : 28x28 image")
print("# Outputs  : 10    (0-9)")

model, x, y, keep_prob = cnn()

# --------------------------------------------------------------
print("==> Setting Up Training")

print("-> Defining loss function")
with tf.name_scope("loss"):
    loss = tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=model)
    loss = tf.reduce_mean(loss)
    tf.summary.scalar("loss", loss)

print("-> Defining optimizer")
with tf.name_scope("optimizer"):
    optimizer = tf.train.AdamOptimizer(learning_rate=1e-3).minimize(loss)

print("-> Defining accuracy")
with tf.name_scope('accuracy'):
    correct_prediction = tf.equal(tf.argmax(model, 1), tf.argmax(y, 1))
    correct_prediction = tf.cast(correct_prediction, tf.float32)
    accuracy = tf.reduce_mean(correct_prediction)
    tf.summary.scalar("accuracy", accuracy)

# --------------------------------------------------------------
sess = tf.Session()

print('==> Defining Tensorboard Writer')
graph_location = tempfile.mkdtemp(prefix='/home/jsb/tmp/')
print('-> Saving graph to: %s' % graph_location)
merged_summary = tf.summary.merge_all()
writer = tf.summary.FileWriter(graph_location)
writer.add_graph(sess.graph)

# --------------------------------------------------------------
print("==> Begin Training")
sess.run(tf.global_variables_initializer())
for i in xrange(20000):
    batch = mnist.train.next_batch(50)
    if i % 100 == 0:
        [train_accuracy, s] = sess.run([accuracy, merged_summary], feed_dict={x: batch[0], y: batch[1], keep_prob: 1.0})
        writer.add_summary(s, i)
        print('step %d, training accuracy %g' % (i, train_accuracy))
    sess.run([optimizer], feed_dict={x: batch[0], y: batch[1], keep_prob: 0.5})

# --------------------------------------------------------------
print("==> Begin Testing")
print('test accuracy %g' % sess.run([accuracy], feed_dict={
    x: mnist.test.images, y: mnist.test.labels, keep_prob: 1.0}))

# --------------------------------------------------------------
print("==> Closing Objects")
writer.close()
sess.close()
