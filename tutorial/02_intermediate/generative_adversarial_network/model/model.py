import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


class GAN(object):
    """ Generative Adversarial Network
    This class is used to create a Generative Adversarial Network (GAN) model, where the Discriminator and Generator
    are both FNN with mirroring hidden sizes (Generator: [h_1, h_2,...,h_n] & Discriminator: [h_n, h_n-1,...,h_1]).
    The random noise used by the Generator is Uniform.

    Authors: Goodfellow, et al.
    Link: https://arxiv.org/abs/1406.2661
    """

    def __init__(self, z_dim, x_dim, hidden_sizes, act_func="lrelu",
                 sess=None, tensor_board=True):
        """Initializes the GAN model.
        Creates the model structure, placeholders, and other parameters.

        Args:
            z_dim (int): The size of the Generator's input used by the Uniform Random Number Generator.
            x_dim (int): The size of the Discriminator's input and the Generator's output.
            hidden_sizes (list of int): The list of sizes for each hidden layer.
            act_func (str): The name of the activation function to be used by the Discriminator and Generator.
                There are several different activation functions that can be used:
                    lrelu
                    relu
                    elu
                    sigmoid
                    tanh
                (Optional: "lrelu")
            sess (tf.Session): If a previous session wants to be use, you may preset it upon initialization.
                If None, then will generate a session upon training.
                (Optional: None)
            tensor_board (bool): If True, will generate TensorBoard summaries and analyze weights and
                biases of the models while training. Otherwise, will simply run the model.
                (Optional: True)
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

        # ================================================
        # Parameters
        # ================================================
        self._act_func = self._ACTIVATIONS[act_func]
        self._tensor_board = tensor_board

        # --------------------------------
        # Discriminator Placeholders
        # --------------------------------
        self._x_dim = x_dim
        self._d_hidden_sizes = hidden_sizes
        self._x = tf.placeholder(tf.float32, [None, x_dim])

        # --------------------------------
        # Generator Placeholders
        # --------------------------------
        self._z_dim = z_dim
        self._g_hidden_sizes = hidden_sizes[::-1]
        self._z = tf.placeholder(tf.float32, [None, z_dim])

        # ================================================
        # Models
        # ================================================
        self._d_vars = None
        self._g_vars = None
        self._d_real = self._discriminator(self._x, x_dim, hidden_sizes, reuse=False)
        self._z = tf.placeholder(tf.float32, [None, z_dim])
        self._g = self._generator(self._z, z_dim, x_dim, self._g_hidden_sizes)
        self._d_fake = self._discriminator(self._g, x_dim, hidden_sizes, reuse=True)

    def __call__(self, samples=1):
        """Generates a sample input that is compared by the discriminator during training.

        Args:
            samples (int or numpy.ndarray): If int, will generate the number of samples with a random uniform sampler.
                If numpy.ndarray, will use the inputted samples as the input for the Generator.

        Returns:
            (tuple of numpy.ndarray) Generated samples and input used.
        """
        return self.generate(samples)

    @staticmethod
    def _weights(shape, name="W"):
        """Used to generate variables used for weights.

        Args:
            shape (list of int): Used to generate a normal tensor.
            name (str): Name used in tf.Variable and tf.summary for TensorBoard

        Returns:
            (tf.Variable). Weight variables.
        """
        initial = tf.random_normal(shape, stddev=tf.sqrt(2. / shape[0]))
        return tf.Variable(initial, name=name)

    @staticmethod
    def _bias(shape, constant=0., name="B"):
        """Used to generate variables used for biases.

        Args:
            shape (list of int): Used to generate a constant tensor.
            constant (float): Used to define all values in the returned variable.
            name (string): Name used in tf.Variable and tf.summary for TensorBoard.

        Returns:
            (tf.Variable). Bias variables.
        """
        initial = tf.constant(constant, shape=shape)
        return tf.Variable(initial, name=name)

    @staticmethod
    def _lrelu(x, alpha=0.01):
        """Leaky Rectified Linear Unit (LRelu)

        Args:
            x (tf.Tensor): A Tensorflow tensor used by the activation function.
            alpha (float): A decimal used to adjust the activation function when the Tensor, x, is negative.
                (Optional: 0.01)

        Returns:
            (tf.Tensor). Output of activation function.
        """
        return tf.nn.relu(x) - alpha * tf.nn.relu(-x)

    def _generator(self, z, input_size, output_size, hidden_sizes):
        """A Generator that is used by the GAN model.

        Args:
            z (tf.Tensor): The input for the Generator, typically a placeholder for later training and sampling.
            input_size (int): The length of the Tensor, z.
            output_size (int): The length of the Discriminator input, x.
            hidden_sizes (list of int): A list of integers, each representing the
                size of a hidden layer used by the Generator.

        Returns:
            (tf.Tensor). A Tensor that matches the input size of the Discriminator.
        """
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
        """A Discriminator that is used by the GAN model.

        Args:
            x (tf.Tensor): The input for the Discriminator, typically a placeholder for
                later training and discriminating.
            input_size (int): The length of the Tensor, z.
            hidden_sizes (list of int): A list of integers, each representing the
                size of a hidden layer used by the Discriminator.
            reuse (bool): If True, will reuse a set of predefined variables

        Returns:
            (tf.Tensor). The output of the Discriminator that is size [n, 1] where n is the number of inputs.
        """
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
    def _plot(samples):
        """Plots given samples taken from the generator during training.

        Args:
            samples (numpy.ndarray): Samples generated by the Generator.

        Returns:
            (matplotlib.Figure). Figure containing the samples.
        """
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

    @staticmethod
    def sample(m, n):
        """Generates random uniform sample input of range [-1, 1] for the Generator.

        Args:
            m (int): Number of samples
            n (int): Size of each sample

        Returns:
            (numpy.ndarray). The samples that are fed into the Generator.
        """
        return np.random.uniform(-1., 1., size=[m, n])

    def train(self, x, learning_rate=1e-3, batch_size=32,
              n_epochs=20000, epoch_print=None, reuse=True):
        """Trains the Discriminator and Generator for (n_epochs * (ceil(x.shape[0] / batch_size))).

        Args:
            x (numpy.ndarray): The real data the Discriminator uses to discriminate between generated inputs.
            learning_rate (float): The learning rate for the Adam Optimizers. (Optional: 1e-3)
            batch_size (int): The number of inputs per each iteration. (Optional: 32)
            n_epochs (int): The number of epochs or cycles through all input data. (Optional: 20000)
            epoch_print (int or None): If None, will not print out any information during training. If int, will
                print out information and plot figures in ratio with the number of data points.
                (i.e. 2 -> prints information in the beginning and middle of each epoch)
                (Optional: None)
            reuse (bool): If True, will reuse the previous created or given session.
                Otherwise, will create a new session. (Optional: True)
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
        if epoch_print:
            if epoch_print and epoch_print <= 0:
                epoch_print = None
            elif int(np.floor(total_batches / epoch_print)) >= 1:
                epoch_print = int(np.floor(total_batches / epoch_print))
            else:
                epoch_print = 1

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
                sample_z = self.sample(batch_size, self._z_dim)
                _, d_loss_curr = self._sess.run([d_optim, d_loss], feed_dict={self._x: batch_x,
                                                self._z: sample_z})
                _, g_loss_curr = self._sess.run([g_optim, g_loss], feed_dict={self._x: batch_x,
                                                self._z: sample_z})

                # Prints Information to Epoch
                if epoch_print and (batch_iter % epoch_print) == 0:
                    print(epoch, batch_iter, total_batches, d_loss_curr, g_loss_curr)
                    samples = self._sess.run(self._g, feed_dict={self._z: self.sample(16, self._z_dim)})

                    fig = self._plot(samples)
                    plt.savefig("out/{}.png"
                                .format(str(file_index).zfill(3)), bbox_inches="tight")
                    file_index += 1
                    plt.close(fig)
                batch_iter += 1

    def generate(self, samples=16):
        """Generates samples from the generator.

        Args:
            samples (int or numpy.ndarray): If int, will generate n random uniform inputs for the Generator and will
                return the generated samples and used inputs. If numpy.ndarray, will use samples parameter as the input
                for the Generator and will return the generated samples and the parameter itself.

        Returns:
            (numpy.ndarray, numpy.ndarray). Two numpy.ndarrays. The first being the generated samples and the second
            being the used input for the Generator.
        """
        if (type(samples) is np.ndarray) and (samples.shape[1] == self._z_dim):
            return self._sess.run(self._g, feed_dict={self._z: samples}), samples
        z = self.sample(samples, self._z_dim)
        return self._sess.run(self._g, feed_dict={self._z: z}), z

    def discriminate(self, x):
        """Discriminates the given input, identifying whether it is real (1) or fake (0).

        Args:
            x (numpy.ndarray): The inputted array that is discriminated by the Discriminator.

        Returns:
            (numpy.ndarray). A [n, 1]-shaped array corresponding with each input and identify how realistic it is
            when compared with the real data and generated data during training.
        """
        return self._sess.run(self._d_real, feed_dict={self._x: x})
