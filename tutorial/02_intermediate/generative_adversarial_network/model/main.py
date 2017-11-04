from model import GAN
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("/tmp/tensorflow/mnist/input_data", one_hot=True)

# Actual maximum: 110000
X, _ = mnist.train.next_batch(110000)
print X.shape

gan = GAN(64, 784, [128])
gan.train(X, batch_print=1000, batch_size=22, n_epochs=34)
