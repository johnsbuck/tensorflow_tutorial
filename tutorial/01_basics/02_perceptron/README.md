# Perceptrons
This section goes over the basic Perceptron model that is used in
several modern Machine Learning models. The main example compares
the model to an analytic Linear Regressor in a basic toy problem
involving the separation of two semi-circles.

## Contents
1. main.py
    * An example script that compares the performance of a single
    Perceptron with an analytical model using the semi-circle problem.
2. models.py
    * Contains two models, Perceptron and LinearRegressor, that are
    used in the example script.
3. data.py
    * Generates two semi-circles consisting of an x,y-coordinate array
    and a label of -1 or +1 depending on which semi-circle the point
    belongs to.