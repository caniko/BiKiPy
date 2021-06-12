import numpy as np


def absolute_derivative(sequence):
    return np.abs(np.diff(sequence, axis=0))
