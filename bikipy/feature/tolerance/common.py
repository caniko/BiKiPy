from logging import getLogger
from typing import Callable

import numpy as np
from numba import njit
from pydantic import validate_arguments
from pydantic_numpy.dtype import NDArrayBool

from bikipy import runtime_settings

logger = getLogger(__name__)


@validate_arguments
def tolerance_model_warning_wrapper(
    tolerance_model: Callable, result_length: int, *args, **kwargs
) -> np.ndarray[bool, bool]:
    attention_boolean_index = tolerance_model(*args, **kwargs)

    if attention_boolean_index is None:
        logger.warning("Tolerance model is completely False")
        return np.zeros(result_length, dtype=bool)

    truth_percentage = len(attention_boolean_index) / np.sum(attention_boolean_index)
    if truth_percentage < 0.01:
        logger.warning(f"Tolerance model has less than 1% True, {truth_percentage}")

    return attention_boolean_index


def common_preparation(
    minimum_seconds_attention: float, maximum_seconds_distraction: float, fps: float, boolean_sequence: NDArrayBool
) -> tuple[float, float, int]:
    return round(minimum_seconds_attention * fps), round(maximum_seconds_distraction * fps), len(boolean_sequence)


if not runtime_settings.disable_numba:
    common_preparation = njit(cache=True)(common_preparation)
