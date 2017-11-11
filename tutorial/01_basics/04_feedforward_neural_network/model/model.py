import numpy as np
import tensorflow as tf
import tempfile


class FNN(object):
    """Feedforward Neural Network
    This class is used to create a Feedforward Neural Network (FNN) model.
    """

    def __init__(self, input_size, output_size, hidden_sizes,
                 act_func="relu", name="FNN", sess=None, tensor_board=True):
        """Initializes the Feedforward Neural Network model.
        Creates the model structure, placeholders, and other parameters.

        Args:
            input_size (int): Size of input for input arrays
            output_size (int): Size of output for model output
            hidden_sizes (list of int): A list of integers, each representing the size of a hidden layer.
            act_func (str): A given activation function name used as part of the model.
                Current options are "relu", "elu", "sigmoid", and "tanh".
                (Optional: "relu")
            name (str): The name of the model. (Optional: "FNN")
            sess (tf.Session): A session to be used for training. If None is given, model will generate a new sess.
                (Optional: None)
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
        self._sess = sess
        self._name = name

        # ================================================
        # Parameters
        # ================================================
        self._in_size = input_size
        self._out_size = output_size
        self._hidden_sizes = hidden_sizes
        self._act_func = self._ACTIVATIONS[act_func]
        self._tensor_board = tensor_board

        # ================================================
        # Define Model
        # ================================================
        self._n_layers = 1
        current = input_size

        with tf.name_scope(self._name):
            self._x = tf.placeholder(tf.float32, [None, input_size])
            self._y = tf.placeholder(tf.float32, [None, output_size])

            # --------------------------------
            # Begin model with initial inputs
            # --------------------------------
            self._model = self._x

            # --------------------------------
            # Define each hidden layer
            # --------------------------------
            for hidden in hidden_sizes:
                with tf.name_scope("fc" + str(self._n_layers)):
                    weights = self._weights([current, hidden])
                    bias = self._bias([hidden])
                    self._model = self._act_func(tf.matmul(self._model, weights) + bias)
                    if tensor_board:
                        tf.summary.histogram("W", weights)
                        tf.summary.histogram("B", bias)
                        tf.summary.histogram("Activation", self._model)
                current = hidden
                self._n_layers += 1

            # --------------------------------
            # Define output layer
            # --------------------------------
            with tf.name_scope("fc" + str(self._n_layers)):
                weights = self._weights([current, output_size])
                bias = self._bias([output_size])
                self._model = self._act_func(tf.matmul(self._model, weights) + bias)
                if tensor_board:
                    tf.summary.histogram("W", weights)
                    tf.summary.histogram("B", bias)
                    tf.summary.histogram("Activation", self._model)

    def __call__(self, x):
        """Returns the predicted output from the model using the given inputs.

        Args:
            x(numpy.ndarray): Inputs that are given to model. Shape: [# of Inputs, Input size defined in object]

        Returns:
            (numpy.ndarray) Output from model. Shape: [# of Inputs, Output size defined in object]
        """
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
            n_epochs(int): Number of iterations. (Optional: 20000)
            epoch_print (int or None): If None, will not print out any information during training. If int, will
                print out information and plot figures in ratio with the number of data points.
                (i.e. 2 -> prints information in the beginning and middle of each epoch)
                (Optional: None)
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
            perm = np.random.permutation(np.arange(x.shape[0]))
            x = x[perm]
            y = y[perm]

            # --------------------------------
            # Training Each Batch
            # --------------------------------
            curr_batch = 0  # Index used to obtain next batch
            batch_iter = 0  # Current iteration of batches

            # Epoch Information
            if epoch_print and (epoch % epoch_print) == 0:
                train_accuracy = self._sess.run(accuracy, feed_dict={self._x: x,
                                                                     self._y: y})
                if self._tensor_board:
                    s = self._sess.run(merged_summary, feed_dict={self._x: x,
                                                                  self._y: y})
                    writer.add_summary(s, batch_iter + (total_batches * epoch))
                print("epoch %d, training accuracy %g" % (epoch, train_accuracy))

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

                # Training Step
                self._sess.run(optimizer, feed_dict={self._x: batch_x,
                                                     self._y: batch_y})
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
        return self._sess.run(self._model, feed_dict={self._x: x})

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

        return self._sess.run(accuracy, feed_dict={self._x: x, self._y: y})
