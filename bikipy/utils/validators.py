from typing import Union, SupportsFloat, Sequence
from warnings import warn
import numpy as np


class TooManyNansError(ValueError):
    def __init__(self, *args, nan_ratio: SupportsFloat = None):
        self.nan_ratio = float(nan_ratio)
        super().__init__(*args)


def compute_nan_ratio(data: Sequence):
    """
    Computes the ratio of data elements that are not NaN and NaN

    NaN is the abbreviation of "not a number"

    :param data: Data sequence
    :return: Ratio of data elements that are not NaN and NaN
    """
    data = np.asanyarray(data)

    if data.dtype == object:
        return 1.0

    return np.sum(np.isnan(data)) / data.size


def validate_nan_ratio(
    nan_ratio: SupportsFloat,
    warn_ratio: SupportsFloat = 0.40,
    error_ratio: Union[SupportsFloat, None] = None,
) -> bool:
    """
    Validate the nan ratio; raise error, warn or do nothing based on parameters

    NaN is the abbreviation of "not a number"

    :param nan_ratio: Float representing the NaN ratio
    :param warn_ratio: Float representing the NaN ratio in which a warning is generated
    :param error_ratio: Float representing the NaN ratio in which an exception is raised
    :return: False if greater than or equal to warn_ratio; True if lesser than warn_ratio
    """
    nan_ratio = float(nan_ratio)

    if error_ratio and nan_ratio >= float(error_ratio):
        msg = (
            f"The data consists of {nan_ratio*100}% NaNs, "
            f"which is too high for the workflow."
            f"It is, therefore, recommended to re-gather the respective data"
        )
        raise TooManyNansError(msg, nan_ratio=nan_ratio)

    elif nan_ratio >= float(warn_ratio):
        warn(f"The data consists of {nan_ratio*100}% NaNs")
        return False
    else:
        return True
