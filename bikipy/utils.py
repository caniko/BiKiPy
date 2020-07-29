from warnings import warn
import numpy as np


def compute_nan_ratio(data):
    data = np.asarray(data)
    return np.sum(np.isnan(data)) / data.size


def validate_nan_ratio(nan_ratio, warn_ratio: float = 0.40, error_ratio: float = 0.95):
    if nan_ratio >= error_ratio:
        msg = (
            f"The data consists of {nan_ratio*100}% NaNs, "
            f"which is too high for the workflow."
            f"It is, therefore, recommended to re-gather the respective data"
        )
        raise ValueError(msg)
    elif nan_ratio >= warn_ratio:
        warn(f"The data consists of {nan_ratio*100}% NaNs")
