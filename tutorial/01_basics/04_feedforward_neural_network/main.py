#!usr/bin/env python2

"""Feedforward Neural Network
This is an example of a feedforward neural network on TensorFlow using MNIST dataset.
"""

from __future__ import print_function
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import tempfile


def weight_variables(shape, name="W"):
    """ Used to generate variables used for weights in the FNN.

    Args:
        shape: List(Integer). Used to generate a normal tensor.
        name: String. Name used in tf.Variable and tf.summary for TensorBoard

    Returns:
        Variable
    """
    initial = tf.random_normal(shape, stddev=tf.sqrt(2. / shape[0]))
    return tf.Variable(initial, name=name)


def bias_variables(shape, constant=0.01, name="B"):
    """ Used to generate variables used for biases in the FNN.

    Args:
        shape: List(Integer). Used to generate a constant tensor.
        constant: A float or integer used to define all values in the returned variable.
        name: Name used in tf.Variable and tf.summary for TensorBoard

    Returns:
        Variable
    """
    initial = tf.constant(constant, shape=shape)
    return tf.Variable(initial, name=name)

def fc_layer(input, in_size, out_size, wname="W", bname="B"):
    """Used to define each layer of our FNN

    Args:
        input: Previous tensor or placeholder used by model
        in_size: Integer. Used to define size of previous output
        out_size: Integer. Used to define size of current output

    Returns:
        Tensor
    """
    W = weight_variables([in_size, out_size], name=wname)
    B = bias_variables([out_size], name=bname)
    tf.summary.histogram(wname, W)
    tf.summary.histogram(bname, B)
    return tf.nn.relu(tf.matmul(input, W) + B)

def fnn():
    """Defines a feedforward neural network and placeholders.

    Returns:
        Tensor
    """
    x = tf.placeholder(tf.float32, [None, 784])  # None used for batch size in training.
    y = tf.placeholder(tf.float32, [None, 10])
    with tf.name_scope("fc1"):  # Used for graphing in tensorboard
        model = fc_layer(x, 784, 500)
    with tf.name_scope("fc2"):
        model = fc_layer(model, 500, 10)
    return model, x, y

# --------------------------------------------------------------
print("==> Importing MNIST")

mnist = input_data.read_data_sets("/tmp/tensorflow/mnist/input_data", one_hot=True)

# --------------------------------------------------------------
print("==> Creating Feedforward Neural Network (FNN) Model")

print("-> Creating placeholders for input & output")
print("# Inputs   : 784   (28x28)")
print("# Outputs  : 10    (0-9)")

print("-> Defining model structure")
model, x, y = fnn()

# --------------------------------------------------------------
print("==> Setting Up Training")

print("-> Defining loss function")
with tf.name_scope("loss"):     # Used for graphing in tensorboard
    loss = tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=model)
    loss = tf.reduce_mean(loss)
    tf.summary.scalar("loss", loss)

print("-> Defining optimizer")
with tf.name_scope("optimizer"):
    optimizer = tf.train.AdamOptimizer(learning_rate=1e-3).minimize(loss)

print("-> Defining accuracy")
with tf.name_scope("accuracy"):
    correct_prediction = tf.equal(tf.argmax(model, 1), tf.argmax(y, 1))
    correct_prediction = tf.cast(correct_prediction, tf.float32)
    accuracy = tf.reduce_mean(correct_prediction)
    tf.summary.scalar("accuracy", accuracy)

print("-> Defining graphing properties")
sess = tf.Session()

graph_location = tempfile.mkdtemp()
print("Saving graph to: %s" % graph_location)

merged_summary = tf.summary.merge_all()

writer = tf.summary.FileWriter(graph_location)
writer.add_graph(sess.graph)

# --------------------------------------------------------------
print("==> Begin Training")
sess.run(tf.global_variables_initializer())
for i in xrange(200):
    batch = mnist.train.next_batch(50)
    if i % 100 == 0:    # Evaluate every 100 batches
        [train_accuracy, s] = sess.run([accuracy, merged_summary], feed_dict={
            x: batch[0], y: batch[1]})
        writer.add_summary(s, i)
        print("step %d, training accuracy %g" % (i, train_accuracy))

    # Training Step
    sess.run(optimizer, feed_dict={x: batch[0], y: batch[1]})

# --------------------------------------------------------------
print("==> Begin Testing")
print("test accuracy %g" % sess.run(accuracy, feed_dict={
    x: mnist.test.images, y: mnist.test.labels}))

# --------------------------------------------------------------
print("==> Closing Objects")
writer.close()
sess.close()
