import numpy as np


def np_abs_diff(sequence):
    return np.abs(np.diff(sequence, axis=0))
