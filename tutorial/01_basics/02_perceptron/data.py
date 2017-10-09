import math
import numpy as np
import random


def double_semi_circle_data(rad=10, thk=5, sep=5, num=2000):
    """ Generates a set of data consisting of two, separated semi-circles.

    Args:
        rad (float): The radius of each semi-circle
        thk (float: The thickness of each semi-circle
        sep (float): The distance between the two mid-points
        num (int): The number of data points

    Returns:
        Two numpy.ndarray containing input and label data.
            X (numpy.ndarray): [num, 2]-shaped array containing x,y-coordinates.
            Y (numpy.ndarray): [num, 1]-shaped array containing labels for X.
    """
    X = []
    Y = []

    midpoint = [(rad + thk / 2) / 2, sep / 2]

    for _ in xrange(num):
        if random.randint(0, 1) == 0:  # Y[n] = -1
            r = rad + random.random() * thk
            theta = random.random() * math.pi
            x_coord = r * math.cos(theta) - midpoint[0]
            y_coord = r * math.sin(theta) + midpoint[1]

            X.append([x_coord, y_coord])
            Y.append([-1])

        else:                           # Y[n] = +1
            r = rad + random.random() * thk
            theta = random.random() * math.pi + math.pi
            x_coord = r * math.cos(theta) + midpoint[0]
            y_coord = r * math.sin(theta) - midpoint[1]

            X.append([x_coord, y_coord])
            Y.append([1])

    return np.array(X), np.array(Y)
