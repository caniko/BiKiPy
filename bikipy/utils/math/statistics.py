from collections.abc import Sequence
from typing import Optional

import numpy as np
from pydantic_numpy import NDArray


def feature_scale(
    data: Sequence,
    real_min: Optional[float] = None,
    real_max: Optional[float] = None,
) -> NDArray:
    """
    Scale the data to [0, 1]; 0 is the smallest and 1 is the highest

    Will find the minimum and maximum from data if no real values are provided

    :param data: Data sequence
    :param real_min: Manual definition of the minimum
    :param real_max: Manual definition of the maximum
    :return: Feature scaled data in nd.array
    """

    data = np.asarray(data)

    minimum = real_min or data.min()
    maximum = real_max or data.max()

    return (data - minimum) / maximum - minimum
