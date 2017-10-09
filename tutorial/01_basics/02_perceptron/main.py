import matplotlib.pyplot as plt
import numpy as np
from models import Perceptron
from models import LinearRegressor as LinearRegressor
from data import double_semi_circle_data


def third_order_polynomial(X):
    """Converts input array to third order polynomial based on custom setup.

    Args:
        X (numpy.ndarray): An [n_samples, 2]-shaped array that is converted to a third-order polynomial.

    Returns:
        (numpy.ndarray) 3rd Order Polynomial array
    """
    return np.array([X[:, 0], X[:, 1], X[:, 0]**2, X[:, 0] * X[:, 1], X[:, 1]**2,
              X[:, 0]**3, X[:, 0]**2 * X[:, 1], X[:, 1]**2 * X[:, 0], X[:, 1]**3]).T


def plot_model_outcomes(X, Y, perc, lin_regressor, title='Perceptron vs. Linear Regressor', third_order=False):
    """Plots outcomes of Perceptron and Linear Regressor model on Meshgrid.

    Args:
        X (numpy.ndarray): A 2-D array consisting of inputs used to plot models.
        Y (numpy.ndarray): A 1-D array consisting of labels used to color the plot based on different labels.
        perc (Perceptron): The Perceptron model used in training.
        lin_regressor (LinearRegressor): The LinearRegressor model used in training.
        title (str): The title of the plot. (Optional: 'Perceptron vs. Linear Regressor')
        third_order (bool): If True, will use 3rd Order Polynomial inputs for prediction instead of 1st Order.
            Otherwise will use 1st Order Polynomial input for prediction.
            (Optional: False)

    Returns:

    """
    # Obtain the indices for each class to separate in plotting
    neg = np.where(Y == -1)[0]
    pos = np.where(Y == 1)[0]

    # --------------------------------
    # Plot Predictions in Meshgrid
    # --------------------------------
    h = .02

    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    # Plot settings
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.autoscale(False)
    plt.xlabel('x2')
    plt.ylabel('x1')

    # Predict values for meshgrid
    mesh_inputs = np.c_[xx.ravel(), yy.ravel()]
    if third_order:
        Z = perc.predict(third_order_polynomial(mesh_inputs))
    else:
        Z = perc.predict(mesh_inputs)
    Z = Z.reshape(xx.shape)

    # Create contour based on Z-label values and x,y-coordinates from mesh
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral, alpha=.25)
    # Create line of separation from meshgrid
    plt.contour(xx, yy, Z, colors='m')

    if third_order:
        Z = lin_regressor.predict(third_order_polynomial(mesh_inputs))
    else:
        Z = lin_regressor.predict(mesh_inputs)
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral, alpha=.25)
    plt.contour(xx, yy, Z, colors='g')

    plt.plot(X[neg, 0], X[neg, 1], 'rx', label='-1')
    plt.plot(X[pos, 0], X[pos, 1], 'bo', label='+1')
    plt.plot([-50, -50], [-50, -50], 'm', label='Perceptron')
    plt.plot([-50, -50], [-50, -50], 'g', label='Linear')

    plt.legend(loc=1)
    plt.title(title)

    plt.show()
    plt.cla()


def train_algorithms(X, Y, perc, lin_regressor):
    """Trains each model using the given inputs and labels.

    Args:
        X (numpy.ndarray): An array of inputs used to train each model.
        Y (numpy.ndarray): A 1-D array of labels used to train each model.
        perc (Perceptron): The Perceptron model that is trained.
        lin_regressor (LinearRegressor): The LinearRegressor model that is trained.

    Returns:
        Three outputs:
            (float) Accuracy of Perceptron
            (float) Accuracy of LinearRegressor
            (list<float>) Progression of Perceptron Accuracy
    """
    # --------------------------------
    # Training Perceptron
    # --------------------------------
    print "=> Training Pocket Algorithm"
    pla_errors = []

    perc.reset(X.shape[1])
    for i in xrange(100000):
        if i % 20000 == 0:
            print str(i / 1000) + '% Complete'
        perc.train_step(X, Y)
        pla_errors.append(perc.accuracy(X, Y))

    print "100% Complete"
    pla_errors.append(perc.accuracy(X, Y))

    # --------------------------------
    # Fitting Linear Regressor
    # --------------------------------
    print "=> Training Linear Regressor"
    lin_regressor.fit(X, Y)

    return perc.accuracy(X, Y), lin_regressor.accuracy(X, Y), pla_errors


def main():
    # ================================================
    # Generating Semi-Circle Data
    # ================================================

    print "==> GENERATE SEMI-CIRCLE DATA"
    X, Y = double_semi_circle_data(sep=-5)

    # ================================================
    # Training 1st Order Polynomial Models
    # ================================================
    print "==> TRAINING WITH 1ST ORDER POLYNOMIAL MODELS"

    # --------------------------------
    # Training Algorithms
    # --------------------------------
    pla_end_errors = []
    lin_end_errors = []

    perc = Perceptron()
    lin_regressor = LinearRegressor()

    perc.reset(X.shape[1])
    perc_end, lin_end, pla_one_errors = train_algorithms(X, Y, perc, lin_regressor)
    pla_end_errors.append(perc_end)
    lin_end_errors.append(lin_end)

    # ================================================
    # Plotting 1st Order Polynomial Models
    # ================================================
    print '==> PLOTTING 1ST ORDER MODELS'
    plot_model_outcomes(X, Y, perc, lin_regressor, title='Perceptron vs. Linear Regressor (1st Order)')

    # ================================================
    # 3rd Order Polynomials
    # ================================================
    print "==> TRAINING WITH 3RD ORDER POLYNOMIAL"

    # Generated 3rd Order Polynomial by creating a simple array.
    print "=> Generating 3rd Order Polynomial Data"
    Z = third_order_polynomial(X)

    # --------------------------------
    # Training Algorithms
    # --------------------------------
    perc = Perceptron()
    lin_regressor = LinearRegressor()

    perc.reset(Z.shape[1])
    perc_end, lin_end, pla_three_errors = train_algorithms(Z, Y, perc, lin_regressor)
    pla_end_errors.append(perc_end)
    lin_end_errors.append(lin_end)

    # ================================================
    # Plotting 3rd Order Polynomial Models
    # ================================================
    print '==> PLOTTING 3RD ORDER MODELS'
    plot_model_outcomes(X, Y, perc, lin_regressor, title='Perceptron vs. Linear Regressor (3rd Order)', third_order=True)

    # ================================================
    # Summary of 1st & 3rd Order Polynomial
    # ================================================

    print "==> LINEAR REGRESSION VS POCKET ALGORITHM (Error)"
    print " Pocket Algorithm"
    print "~~~~~~~~~~~~~~~~~~"
    print "1st Order:", pla_end_errors[0]
    print "3rd Order:", pla_end_errors[1]
    print ""
    print " Linear Regressor"
    print "~~~~~~~~~~~~~~~~~~"
    print "1st Order:", lin_end_errors[0]
    print "3rd Order:", lin_end_errors[1]
    print ""

    print "==> PLOTTING DATA"
    print "=> Creating Line Plot for Pocket Algorithms"
    plt.plot(pla_three_errors, label="3rd Order")
    plt.plot(pla_one_errors, label="1st Order")

    plt.xlabel("Iterations")
    plt.ylabel("Error (%)")
    plt.title('Perceptron Pocket Algorithm (1st vs 3rd Order)')
    plt.legend(loc=1)

    plt.show()
    plt.close()

    print "=> Creating Bar Plot"
    plt.bar(1, pla_end_errors[0], color='#FFB000', label='Pocket Algorithm')
    plt.bar(1, lin_end_errors[0], color='#0066FF', label='Linear Regression')
    plt.bar(2, pla_end_errors[1], color='#FFB000')
    plt.bar(2, lin_end_errors[1], color='#0066FF')

    plt.xticks([1, 2], ('1st', '3rd'))
    plt.xlabel('Order Polynomial')
    plt.ylabel('Error (%)')
    plt.title('Linear Regression vs Perceptron Pocket Algorithm')

    plt.legend(loc=1)
    plt.show()


if __name__ == '__main__':
    main()
