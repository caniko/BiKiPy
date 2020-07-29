from warnings import warn
from typing import Union, SupportsFloat
import numpy as np


class TooManyNansError(ValueError):
    def __init__(self, *args, nan_ratio: SupportsFloat = None):
        self.nan_ratio = nan_ratio
        super().__init__(*args)


def compute_nan_ratio(data):
    data = np.asanyarray(data)

    if data.dtype == object:
        return 1.0

    return np.sum(np.isnan(data)) / data.size


def validate_nan_ratio(
    nan_ratio, warn_ratio: float = 0.40, error_ratio: Union[float, None] = None
) -> bool:
    if error_ratio and nan_ratio >= error_ratio:
        msg = (
            f"The data consists of {nan_ratio*100}% NaNs, "
            f"which is too high for the workflow."
            f"It is, therefore, recommended to re-gather the respective data"
        )
        raise TooManyNansError(msg, nan_ratio=nan_ratio)

    elif nan_ratio >= warn_ratio:
        warn(f"The data consists of {nan_ratio*100}% NaNs")
        return False
    else:
        return True
