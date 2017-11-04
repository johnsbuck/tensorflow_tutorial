import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


class GAN(object):
    """

    """

    def __init__(self, g_input_size, d_input_size, hidden_sizes,
                 act_func="lrelu", sess=None, tensor_board=True,
                 name="GAN"):
        """

        Args:
            g_input_size:
            d_input_size:
            hidden_sizes:
            act_func:
            sess:
            tensor_board:
            name:
        """
        # ================================================
        # Constants
        # ================================================
        self._ACTIVATIONS = {"lrelu": self._lrelu,
                             "relu": tf.nn.relu,
                             "elu": tf.nn.elu,
                             "sigmoid": tf.nn.sigmoid,
                             "tanh": tf.nn.tanh}

        # ================================================
        # Utilities
        # ================================================
        self._sess = sess
        self._name = name

        # ================================================
        # Parameters
        # ================================================
        self._act_func = self._ACTIVATIONS[act_func]
        self._tensor_board = tensor_board

        # --------------------------------
        # Discriminator Placeholders
        # --------------------------------
        self._d_in_size = d_input_size
        self._d_hidden_sizes = hidden_sizes
        self._x = tf.placeholder(tf.float32, [None, d_input_size])

        # --------------------------------
        # Generator Placeholders
        # --------------------------------
        self._g_in_size = g_input_size
        self._g_out_size = d_input_size
        self._g_hidden_sizes = hidden_sizes[::-1]
        self._z = tf.placeholder(tf.float32, [None, g_input_size])

        # ================================================
        # Models
        # ================================================

        self._d_vars = None
        self._g_vars = None
        self._d_real = self._discriminator(self._x, d_input_size, hidden_sizes, reuse=False)
        self._z = tf.placeholder(tf.float32, [None, g_input_size])
        self._g = self._generator(self._z, g_input_size, d_input_size, self._g_hidden_sizes)
        self._d_fake = self._discriminator(self._g, d_input_size, hidden_sizes, reuse=True)

    def __call__(self, x=None):
        return self.generate(x)

    @staticmethod
    def _weights(shape, name="W"):
        """ Used to generate variables used for weights.

        Args:
            shape (list<int>): Used to generate a normal tensor.
            name (str): Name used in tf.Variable and tf.summary for TensorBoard

        Returns:
            (tf.Variable). Weight variables.
        """
        initial = tf.random_normal(shape, stddev=tf.sqrt(2. / shape[0]))
        return tf.Variable(initial, name=name)

    @staticmethod
    def _bias(shape, constant=0., name="B"):
        """ Used to generate variables used for biases.

        Args:
            shape (list<int>): Used to generate a constant tensor.
            constant (float): Used to define all values in the returned variable.
            name (string): Name used in tf.Variable and tf.summary for TensorBoard

        Returns:
            (tf.Variable). Bias variables
        """
        initial = tf.constant(constant, shape=shape)
        return tf.Variable(initial, name=name)

    @staticmethod
    def _lrelu(x, alpha=0.01):
        return tf.nn.relu(x) - alpha * tf.nn.relu(-x)

    def _generator(self, z, input_size, output_size, hidden_sizes):
        with tf.variable_scope("generator"):
            self._g_vars = []
            g = z
            n_layer = 1

            current = input_size
            for layer in hidden_sizes:
                with tf.name_scope("fc" + str(n_layer)):
                    W = self._weights([current, layer])
                    b = self._bias([layer])
                    self._g_vars += [W, b]
                    g = self._act_func(tf.matmul(g, W) + b)
                    current = layer
                    n_layer += 1

            with tf.name_scope("fc" + str(n_layer)):
                W = self._weights([current, output_size])
                b = self._bias([output_size])
                self._g_vars += [W, b]
                return tf.nn.sigmoid(tf.matmul(g, W) + b)

    def _discriminator(self, x, input_size, hidden_sizes, reuse=False):
        with tf.variable_scope("discriminator"):
            new_vars = False
            if (not reuse) or (self._d_vars is None):
                self._d_vars = []
                new_vars = True
            d = x
            n_layers = 1

            current = input_size
            for layer in hidden_sizes:
                with tf.name_scope("fc" + str(n_layers)):
                    if new_vars:
                        W = self._weights([current, layer])
                        b = self._bias([layer])
                        self._d_vars += [W, b]
                    d = self._act_func(tf.matmul(d, self._d_vars[2*(n_layers-1)]) + self._d_vars[2*(n_layers-1)+1])
                    current = layer
                    n_layers += 1

            with tf.name_scope("fc" + str(n_layers)):
                if new_vars:
                    W = self._weights([current, 1])
                    b = self._bias([1])
                    self._d_vars += [W, b]
                return tf.matmul(d, self._d_vars[-2]) + self._d_vars[-1]

    @staticmethod
    def sample(m, n):
        return np.random.uniform(-1., 1., size=[m, n])

    @staticmethod
    def plot(samples):
        fig = plt.figure(figsize=(4, 4))
        gs = gridspec.GridSpec(4, 4)
        gs.update(wspace=0.05, hspace=0.05)

        for i, sample in enumerate(samples):
            ax = plt.subplot(gs[i])
            plt.axis("off")
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_aspect("equal")
            plt.imshow(sample.reshape(28, 28), cmap="Greys_r")

        return fig

    def train(self, x, learning_rate=1e-3, batch_size=32,
              n_epochs=20000, batch_print=None, reuse=True):
        """

        Args:
            x:
            learning_rate:
            batch_size:
            n_epochs:
            batch_print:
            reuse:

        Returns:

        """
        with tf.name_scope("d_loss"):
            with tf.name_scope("d_real_loss"):
                d_real_loss = tf.reduce_mean(
                         tf.nn.sigmoid_cross_entropy_with_logits(
                             logits=self._d_real,
                             labels=tf.ones_like(self._d_real)))
            with tf.name_scope("d_fake_loss"):
                d_fake_loss = tf.reduce_mean(
                    tf.nn.sigmoid_cross_entropy_with_logits(
                        logits=self._d_fake,
                        labels=tf.zeros_like(self._d_fake)))
            d_loss = d_real_loss + d_fake_loss

        with tf.name_scope("g_loss"):
            g_loss = tf.reduce_mean(
                     tf.nn.sigmoid_cross_entropy_with_logits(logits=self._d_fake,
                                                             labels=tf.ones_like(self._d_fake)))

        with tf.name_scope("d_optimizer"):
            d_optim = tf.train.AdamOptimizer(learning_rate).minimize(d_loss, var_list=self._d_vars)
        with tf.name_scope("g_optimizer"):
            g_optim = tf.train.AdamOptimizer(learning_rate).minimize(g_loss, var_list=self._g_vars)

        if (not reuse) or (self._sess is None):
            if self._sess is not None:
                self._sess.close()
            self._sess = tf.Session()
            self._sess.run(tf.global_variables_initializer())

        # ================================================
        # Begin Training
        # ================================================
        total_batches = int(np.ceil(x.shape[0] * 1. / batch_size))

        file_index = 0
        for epoch in range(n_epochs):
            # --------------------------------
            # Randomize Inputs and Outputs
            # --------------------------------
            perm = np.random.permutation(np.arange(x.shape[0]))
            x = x[perm]

            # --------------------------------
            # Training Each Batch
            # --------------------------------
            curr_batch = 0  # Index used to obtain next batch
            batch_iter = 0  # Current iteration of batches

            while curr_batch < x.shape[0]:

                # Get Data Batch
                if (curr_batch + batch_size) < x.shape[0]:
                    batch_x = x[curr_batch:curr_batch + batch_size]
                    curr_batch += batch_size
                else:
                    batch_x = x[curr_batch:]
                    curr_batch = x.shape[0]

                # Training Step
                sample_z = self.sample(batch_size, self._g_in_size)
                _, d_loss_curr = self._sess.run([d_optim, d_loss], feed_dict={self._x: batch_x,
                                                self._z: sample_z})
                _, g_loss_curr = self._sess.run([g_optim, g_loss], feed_dict={self._x: batch_x,
                                                self._z: sample_z})

                if (batch_print is not None) and ((batch_iter % batch_print) == 0):
                    print(epoch, batch_iter, total_batches, d_loss_curr, g_loss_curr)
                    samples = self._sess.run(self._g, feed_dict={self._z: self.sample(16, self._g_in_size)})

                    fig = self.plot(samples)
                    plt.savefig("out/{}.png"
                                .format(str(file_index).zfill(3)), bbox_inches="tight")
                    file_index += 1
                    plt.close(fig)
                batch_iter += 1

    def generate(self, num_samples=16, x=None):
        if (x is not None) and (x.shape[1] == self._g_in_size):
            return self._sess.run(self._g, feed_dict={self._z: x}), x
        z = self.sample(num_samples, self._g_in_size)
        return self._sess.run(self._g, feed_dict={self._z: z}), z

    def discriminate(self, x):
        return self._sess.run(self._d_real, feed_dict={self._x: x})
