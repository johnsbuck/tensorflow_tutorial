import numpy as np


class Perceptron(object):
    """Perceptron Model
    A basic model used for machine learning.
    Consists iterative training a set of weights used to be used for generating a linear model.
    This Perceptron is focused mainly on classification problems, as it is based on Hebbian learning.
    """

    def __init__(self):
        self._w = None
        self._b = 0.0
        self.__t_w = None
        self.__t_b = 0.0

    def train_step(self, X, y):
        """A single training step for the Perceptron algorithm (based on Pocket Algorithm)

        Args:
            X (numpy.ndarray): Input array used for training model.
            y (numpy.ndarray): Labels used to classify inputs and compare with model output labels.

        Returns:
            (bool) If True, there are no misclassified inputs and training is complete.
                Otherwise, training isn't complete and a training step occurs.
        """
        dot_prod = np.dot(X, self.__t_w) + self._b
        misclassified = np.where(y != np.sign(dot_prod))[0]
        if misclassified.shape[0] == 0:
            return True     # Training complete.
        index = np.random.randint(0, misclassified.shape[0], 1)
        self.__t_w += (y[misclassified[index]] * X[misclassified[index]]).T
        self.__t_b += y[misclassified[index]]
        if self.__training_accuracy(X, y) > self.accuracy(X, y):
            self._w = np.copy(self.__t_w)
            self._b = self.__t_b
        return False

    def reset(self, n_features):
        """Resets the Perceptron to have weights and bias of zero.

        Args:
            n_features (int): Number of features in weights vector.

        Returns:

        """
        self._w = np.zeros((n_features, 1), dtype=np.float64)
        self._b = 0.0
        self.__t_w = np.copy(self._w)
        self.__t_b = self._b

    def train(self, X, y, num_iter=None, reset=True):
        """Runs through several training steps for n iterations or until there are no misclassifications.

        Args:
            X (numpy.ndarray): Input array used for training.
            y (numpy.ndarray): Labels used to train Perceptron.
            num_iter (int): The number of training steps taken by Perceptron.
                If None, will train until there are no misclassifications.
            reset (bool): If True, will reset weight vector and bias.
                Otherwise will keep training with previous parameterss.

        Returns:
            (int) Number of iterations for training.
        """
        n_samples, n_features = X.shape
        if reset or self._w is None:
            self._w = np.zeros((n_features, 1), dtype=np.float64)
            self._b = 0.0
        self.__t_w = np.copy(self._w)
        self.__t_b = self._b

        trained = False
        iterations = 0
        while not trained:
            if num_iter is not None and iterations == num_iter:
                return iterations
            trained = self.__train_step(X, y)
            iterations += 1
        return iterations

    def predict(self, X):
        """ Uses Perceptron to predict classification of inputs.

        Args:
            X (numpy.ndarray): An input array that is given to Perceptron from classification.

        Returns:
            (numpy.ndarray) Classifications for each label {-1, 0, +1}.
        """
        if self._w is not None:
            return np.sign(np.dot(X, self._w) + self._b)
        raise Exception("Weights not initialized. Perceptron needs to be trained.")

    def accuracy(self, X, y):
        """ Used to measure the accuracy of the Perceptron.

        Args:
            X (numpy.ndarray): Input array used for predicting accuracy.
            y (numpy.ndarray): Labels used to compare with the Perceptron results for accuracy.

        Returns:
            (float) # of correct Inputs / # of all Inputs
        """
        if self._w is None:
            raise Exception("Weights not initialized. Perceptron needs to be trained.")

        dot_prod = np.dot(X, self._w) + self._b
        correct = np.where(y == np.sign(dot_prod))[0]
        return correct.shape[0]/float(X.shape[0])

    def __training_accuracy(self, X, y):
        """ Used to measure the accuracy of the training weights.

        Args:
            X (numpy.ndarray): Input array used for predicting accuracy.
            y (numpy.ndarray): Labels used to compare with the Perceptron results for accuracy.

        Returns:
            (float) # of correct Inputs / # of all Inputs
        """
        if self.__t_w is None:
            raise Exception("Weights not initialized. Perceptron needs to be trained.")

        dot_prod = np.dot(X, self.__t_w) + self.__t_b
        correct = np.where(y == np.sign(dot_prod))[0]
        return correct.shape[0]/float(X.shape[0])


class LinearRegressor(object):
    """Linear Regressor Model
    A basic model used for machine learning.
    Uses the pseudo-inverse of a set of data to calculate the weights of the model.
    """
    def __init__(self):
        self._w = None
        self._b = 0.0

    def fit(self, X, y):
        """Fits the model using the inputs and label given.

        Args:
            X (numpy.ndarray): An array of inputs used to obtain the weights.
            y (numpy.ndarray): An array of labels used to obtain the weights

        Returns:

        """
        X = np.hstack((np.ones((X.shape[0], 1)), X))    # Biases
        self._w = np.dot(np.linalg.pinv(X), y)
        self._w, self._b = self._w[1:], self._w[0]

    def predict(self, X):
        """Predicts the labels of a given input array using the Linear Regressor.

        Args:
            X (numpy.ndarray): An array of inputs whose labels are calculated by the model.

        Returns:
            (numpy.ndarray) An array of labels, each in the set of {-1, 0, +1}.
        """
        return np.sign(np.dot(X, self._w) + self._b)

    def accuracy(self, X, y):
        """ Used to measure the accuracy of the Linear Regressor.

        Args:
            X (numpy.ndarray): Input array used for predicting accuracy.
            y (numpy.ndarray): Labels used to compare with the Linear Regression results for accuracy.

        Returns:
            (float) # of correct Inputs / # of all Inputs
        """
        dot_prod = np.dot(X, self._w) + self._b
        correct = np.where(y == np.sign(dot_prod))[0]
        return correct.shape[0]/float(X.shape[0])
