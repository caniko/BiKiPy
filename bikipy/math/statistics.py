from typing import Optional, Sequence

import numpy as np


def feature_scale(
    data: Sequence,
    real_min: Optional[float],
    real_max: Optional[float],
) -> NpNDArrayFp64:
    """
    Scale the data to [0, 1]; 0 is the smallest and 1 is the highest

    Will find the minimum and maximum from data if no real values are provided

    :param data: Data sequence
    :param real_min: Manual definition of the minimum
    :param real_max: Manual definition of the maximum
    :return: Feature scaled data in nd.array
    """

    data_array = np.asarray(data)

    minimum = real_min or data_array.min()
    maximum = real_max or data_array.max()

    return (data_array - minimum) / maximum - minimum


def nan_average(data, weights):
    ma = np.ma.MaskedArray(data, mask=np.isnan(data))
    return np.ma.average(ma, weights=weights)
