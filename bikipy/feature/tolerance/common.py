from logging import getLogger
from typing import Callable

import numpy as np
from pydantic import validate_arguments

from bikipy.core.typing import NDArrayBool

logger = getLogger(__name__)


@validate_arguments
def tolerance_filter_warning_wrapper(tolerance_filter: Callable, result_length: int, *args, **kwargs) -> NDArrayBool:
    attention_boolean_index = tolerance_filter(*args, **kwargs)

    if attention_boolean_index is None:
        logger.warning("Tolerance filter is completely False")
        return np.zeros(result_length, dtype=bool)

    truth_percentage = len(attention_boolean_index) / np.sum(attention_boolean_index)
    if truth_percentage < 0.01:
        logger.warning(f"Tolerance filter has less than 1% True, {truth_percentage}")

    return attention_boolean_index
