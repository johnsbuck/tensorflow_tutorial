import tensorflow as tf
import numpy as np
import tempfile
import functools
import operator

class CNN(object):
    """Convolutional Neural Network
    This class uses TensorFlow to create a CNN model that is used for regression, logistic regression,
    and classification problems.

    """

    def __init__(self, input_size, output_size, kernel_sizes, fc_hidden_sizes, strides=[1, 1, 1, 1],
                 padding="SAME", chan=1, pool=True, dropout=True, act_func="relu", name="CNN", tensor_board=True):
        """Initializes the Convolutional Neural Network model.
        Creates the model structure, placeholders, and other parameters.

        Args:
            input_size (int or list of int): Size of input for input arrays.
                If just an integer, will assume is a square, 1D input and will reshape accordingly.
                If is a list of integers, the model will not reshape.
            output_size (int): Size of output for model output
            kernel_sizes (list of list of int): A list containing lists of integers, each representing
                a kernel of a convolutional layer.
            fc_hidden_sizes (list of int): A list of integers, each representing
                the size of a fully-connected hidden layer.
            strides (list of int): A list of integers, representing the strides used in each convolutional layer.
                (Optional: [1, 1, 1, 1])
            padding (str): The padding option for the convolutional layer. The padding may be "SAME" or "VALID".
                (Optional: "SAME")
            chan (int): The number of channels.
                Grayscale: 1
                RGB: 3
                RGBA: 4
                (Optional: 1)
            pool (bool): If True, the CNN will pool each convolutional layer with a max-pool of size 2. (Optional: True)
            dropout (bool): If True, will add a dropout layer after the first fully-connected layer. (Optional: True)
            act_func (str): A given activation function name used as part of the model.
                Current options are "relu", "elu", "sigmoid", and "tanh".
                (Optional: "relu")
            name (str): The name of the model. (Optional: "FNN")
            tensor_board (bool): If True, will create name scope to be used for TensorBoard.
                Otherwise, will just create model.
                (Optional: True)
        """
        # ================================================
        # Constants
        # ================================================
        self._ACTIVATIONS = {"relu": tf.nn.relu,
                             "elu": tf.nn.elu,
                             "sigmoid": tf.nn.sigmoid,
                             "tanh": tf.nn.tanh}

        # ================================================
        # Utilities
        # ================================================
        self._sess = None
        self._name = name

        # ================================================
        # Parameters
        # ================================================
        self._in_size = input_size
        self._out_size = output_size
        self._kernel_sizes = kernel_sizes
        self._fc_hidden_sizes = fc_hidden_sizes
        self._strides = strides
        self._padding = padding
        self._chan = chan
        self._pool = pool
        self._dropout = dropout
        self._act_func = self._ACTIVATIONS[act_func]
        self._tensor_board = tensor_board

        # ================================================
        # Define Model
        # ================================================
        n_conv_layers = 0
        n_fc_layers = 1     # +1 for the last output layer.

        with tf.name_scope(self._name):
            if type(input_size) is not list:
                self._x = tf.placeholder(tf.float32, [None, input_size])
            else:
                self._x = tf.placeholder(tf.float32, [None] + input_size)
            self._y = tf.placeholder(tf.float32, [None, output_size])

            # --------------------------------
            # Begin model with initial inputs
            # --------------------------------
            self._model = self._x
            if type(input_size) is not list:
                # If 1D, will reshape based one sqrt of input_size (assumes square image or input)
                self._model = tf.reshape(self._model, [-1, int(tf.sqrt(input_size)), int(tf.sqrt(input_size)), chan])

            # --------------------------------
            # Define each convolutional layer
            # --------------------------------
            for hidden in kernel_sizes:
                with tf.name_scope("conv" + str(n_conv_layers + 1)):
                    weights = self._weights([hidden])
                    bias = self._bias([hidden[-1]])
                    self._model = self._act_func(tf.nn.conv2d(self._model, weights,
                                                              strides=strides, padding=padding) + bias)
                    if tensor_board:
                        tf.summary.image("Filters", weights[:, :, :, :1], max_outputs=1)
                        tf.summary.histogram("W", weights)
                        tf.summary.histogram("B", bias)
                if pool:
                    with tf.name_scope("pool" + str(n_conv_layers)):
                        self._model = tf.nn.max_pool(self._model, ksize=[1, 2, 2, 1],
                                                     strides=[1, 2, 2, 1], padding="SAME")
                        if tensor_board:
                            tf.summary.image("Max-Pooled (2) Filters", self._model[:, :, :, :1], max_outputs=1)
                n_conv_layers += 1

            # --------------------------------
            # Flatten for fc layers
            # --------------------------------
            self._model = tf.reshape(self._model, [-1, functools.reduce(operator.mul, self._model.get_shape()[1:], 1)])

            # --------------------------------
            # Define each fc layer
            # --------------------------------
            current = self._model.get_shape()[1]
            for hidden in fc_hidden_sizes:
                with tf.name_scope("fc" + str(n_fc_layers)):
                    weights = self._weights([current, hidden])
                    bias = self._bias([hidden])
                    self._model = self._act_func(tf.matmul(self._model, weights) + bias)
                current = hidden
                n_fc_layers += 1
                if tensor_board:
                    tf.summary.histogram("W", weights)
                    tf.summary.histogram("B", bias)
                    tf.summary.histogram("Activation", self._model)

                if dropout:
                    with tf.name_scope("dropout"):
                        self._keep_prob = tf.placeholder(tf.float32)
                        self._model = tf.nn.dropout(self._model, self._keep_prob)
                        if tensor_board:
                            tf.summary.histogram("dropout", self._model)
                        dropout = False

            # --------------------------------
            # Define output layer
            # --------------------------------
            with tf.name_scope("fc" + str(n_fc_layers)):
                weights = self._weights([current, output_size])
                bias = self._bias([output_size])
                self._model = self._act_func(tf.matmul(self._model, weights) + bias)
                if tensor_board:
                    tf.summary.histogram("W", weights)
                    tf.summary.histogram("B", bias)
                    tf.summary.histogram("Activation", self._model)
            self._n_layers = n_conv_layers + n_fc_layers

    def __call__(self, x):
        return self.predict(x)

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
    def _bias(shape, constant=0.01, name="B"):
        """Used to generate variables used for biases.

        Args:
            shape (list of int): Used to generate a constant tensor.
            constant (float): Used to define all values in the returned variable.
            name (string): Name used in tf.Variable and tf.summary for TensorBoard

        Returns:
            (tf.Variable). Bias variables
        """
        initial = tf.constant(constant, shape=shape)
        return tf.Variable(initial, name=name)

    def train(self, x, y, learning_rate=1e-3, batch_size=10,
              n_epochs=20000, epoch_print=None, reuse=True):
        """Trains the model using the given input and output.
        Optimizer: AdamOptimizer
        Loss:
            Classification - Averaged Softmax with Logits
            Regression - Mean Squared Error (MSE)
        Accuracy:
            Classification - Correct Classifications / All Classifications
            Regression - Mean Absolute Error (MAE)

        Args:
            x(numpy.ndarray): Input NumPy ndarray used for training model.
            y(numpy.ndarray): Output regression or labels from NumPy ndarray used for training model.
            learning_rate(float): Learning rate used by the AdamOptimizer. (Optional: 1e-3)
            batch_size(int): Size of each batch. (Optional: 10)
            n_iter(int): Number of iterations. (Optional: 20000)
            batch_print(int): Prints out training information every nth batch,
                or never if set to None. (Optional: None)
            iter_mode(str): There are two modes, epochs and batches.
                epochs: Will run through all batches for <n_iter> epochs, randomizing every epochs.
                batches: Will run through <n_iter> batches,
                    randomizing after running through all batches in a given permutation.
                (Optional: "epochs")
            reuse(bool): If True, will reset the session from last. Otherwise, will only set if not defined.
                (Optional: True)

        Returns:

        """
        # ================================================
        # Checking for Input Errors
        # ================================================

        # --------------------------------
        # Errors for input and output
        # --------------------------------

        # Value Errors (shape doesn"t align)
        if x.shape[0] != y.shape[0]:
            raise ValueError("Must have the same number of inputs as outputs.")
        if x.shape[1] != self._in_size:
            raise ValueError("Input doesn\"t match model input size.")
        if y.shape[1] != self._out_size:
            raise ValueError("Output doesn\"t match model output size.")

        # ================================================
        # Define Utilities
        # ================================================

        # --------------------------------
        # Define Training Parameters
        # --------------------------------
        with tf.name_scope("loss"):
            # Classification (Softmax)
            if self._out_size > 1:
                loss = tf.nn.softmax_cross_entropy_with_logits(labels=self._y, logits=self._model)
                loss = tf.reduce_mean(loss)
            # Regression (MSE)
            else:
                loss = tf.reduce_mean(tf.pow(self._y - self._model, 2))
            if self._tensor_board:
                tf.summary.scalar("loss", loss)

        with tf.name_scope("optimizer"):
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

        with tf.name_scope("accuracy"):
            # Classification (Correct Classifications / All Classifications)
            if self._out_size > 1:
                correct_prediction = tf.equal(tf.argmax(self._model, 1), tf.argmax(self._y, 1))
                correct_prediction = tf.cast(correct_prediction, tf.float32)
                accuracy = tf.reduce_mean(correct_prediction)
            # Regression (MAE)
            else:
                accuracy = tf.reduce_mean(tf.abs(self._y - self._model))
            if self._tensor_board:
                tf.summary.scalar("accuracy", accuracy)

        # --------------------------------
        # Create Session
        # --------------------------------
        if (not reuse) or (self._sess is None):
            if self._sess is not None:
                self._sess.close()
            self._sess = tf.Session()
            # Initialize variables for new session
            self._sess.run(tf.global_variables_initializer())

        if self._tensor_board:
            merged_summary = tf.summary.merge_all()
            graph_location = tempfile.mkdtemp()
            writer = tf.summary.FileWriter(graph_location)
            writer.add_graph(self._sess.graph)

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

        for epoch in range(n_epochs):

                # --------------------------------
                # Randomize Inputs and Outputs
                # --------------------------------
                perm = np.arange(x.shape[0])
                x = x[perm]
                y = y[perm]

                # --------------------------------
                # Training Each Batch
                # --------------------------------
                curr_batch = 0  # Index used to obtain next batch
                batch_iter = 0  # Current iteration of batches
                while curr_batch < x.shape[0]:

                    # Get Data Batch
                    if (curr_batch + batch_size) < x.shape[0]:
                        batch_x = x[curr_batch:curr_batch + batch_size]
                        batch_y = y[curr_batch:curr_batch + batch_size]
                        curr_batch += batch_size
                    else:
                        batch_x = x[curr_batch:]
                        batch_y = y[curr_batch:]
                        curr_batch = x.shape[0]

                    # Batch Information
                    if epoch_print and (batch_iter % epoch_print) == 0:
                        train_accuracy = self._sess.run(accuracy, feed_dict={self._x: batch_x,
                                                                             self._y: batch_y,
                                                                             self._keep_prob: 1.0})
                        if self._tensor_board:
                            s = self._sess.run(merged_summary, feed_dict={self._x: batch_x,
                                                                          self._y: batch_y,
                                                                          self._keep_prob: 1.0})
                            writer.add_summary(s, batch_iter + (total_batches * iter))
                        print("step %d, training accuracy %g" % (batch_iter, train_accuracy))

                    # Training Step
                    self._sess.run(optimizer, feed_dict={self._x: batch_x, self._y: batch_y, self._keep_prob: 0.5})
                    batch_iter += 1

    def predict(self, x):
        """Returns the output of the model from the given input array.
        Loss:
            Classification - Averaged Softmax with Logits
            Regression - Mean Squared Error (MSE)

        Args:
            x (numpy.ndarray): Array that is inputted into the model.

        Returns:
            (numpy.ndarray). Predicted output generated by the model.
        """
        return self._sess.run(self._model, feed_dict={self._x: x, self._keep_prob: 1.0})

    def accuracy(self, x, y):
        """Returns the accuracy of the model based on the given input and output arrays.
        Accuracy:
            Classification - Correct Classifications / All Classifications
            Regression - Mean Absolute Error (MAE)

        Args:
            x (numpy.ndarray): Input array that is used to measure model"s accuracy.
            y (numpy.ndarray): Output array that is compared with model"s output calculate accuracy.

        Returns:
            (float). The accuracy of the model based on the actual output.
        """

        # Classification (Correct Classifications / All Classifications)
        if self._out_size > 1:
            correct_prediction = tf.equal(tf.argmax(self._model, 1), tf.argmax(self._y, 1))
            correct_prediction = tf.cast(correct_prediction, tf.float32)
            accuracy = tf.reduce_mean(correct_prediction)

        # Regression (MAE)
        else:
            accuracy = tf.reduce_mean(tf.abs(self._y - self._model))

        return self._sess.run(accuracy, feed_dict={self._x: x, self._y: y, self._keep_prob: 1.0})
