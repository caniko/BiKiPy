from typing import Union, Sequence
import numpy as np


def feature_scale(
    data: Sequence,
    real_min: Union[float, int, None] = None,
    real_max: Union[float, int, None] = None,
):
    """
    Scale the data to [0, 1]; 0 is the smallest and 1 is the highest

    Will find the minimum and maximum from data if no real values are provided

    :param data: Data sequence
    :param real_min: Manual definition of the minimum
    :param real_max: Manual definition of the maximum
    :return:
    """
    data = np.asanyarray(data)

    minimum = real_min or data.min()
    maximum = real_max or data.max()

    return (data - minimum) / maximum - minimum
