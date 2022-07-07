import numpy as np
from numba import njit
from pydantic import validate_arguments

from bikipy import ENABLE_NUMBA
from bikipy.core.typing import NDArrayBool
from bikipy.feature.tolerance import GENERIC_MINIMUM_SECONDS_ATTENTION, GENERIC_MAXIMUM_SECONDS_DISTRACTION
from bikipy.feature.tolerance.common import tolerance_filter_warning_wrapper


@validate_arguments
def single_node_tolerance_filter(
    boolean_index: NDArrayBool,
    fps: float,
    minimum_seconds_attention: float = GENERIC_MINIMUM_SECONDS_ATTENTION,
    maximum_seconds_distraction: float = GENERIC_MAXIMUM_SECONDS_DISTRACTION,
) -> NDArrayBool | None:
    return tolerance_filter_warning_wrapper(
        _filter, len(boolean_index), boolean_index, fps, minimum_seconds_attention, maximum_seconds_distraction
    )


def _filter(
    boolean_index: NDArrayBool,
    fps: float,
    minimum_seconds_attention: float,
    maximum_seconds_distraction: float,
) -> NDArrayBool | None:
    """
    Filters boolean_index with respect to attention. The filter tolerates distraction, and requires
    minimum_seconds_attention to be fulfilled before accepting the sequence as attention.

    1. Including an attention event requires attention time to be greater than minimum_seconds_attention
    2. During an attention event, the subject may be distracted for maximum_seconds_distraction seconds.
        This triggers a sub event:
            a) The subject has to be attentive for minimum_seconds_attention to merge the gap between the new attention
            with the previous.

            b) The events will remain if the distraction time surpasses the maximum_seconds_distraction.
            Note that the new attention might be removed if it is shorter than minimum_seconds_attention

    start is set to 0 when in fact it should be None to support njit mode in numba.

    :param boolean_index:
    :param fps: Frames per second (fps) of the recording used to generate the data in boolean_index
    :param minimum_seconds_attention: Minimum number of seconds that the sequence has to be True
    for it to be defined as an attention sequence. Filtered sequences will be converted to False.
    :param maximum_seconds_distraction:
    :type boolean_index: NDArrayFp64
    :type fps: float
    :type minimum_seconds_attention: float
    :return: Boolean index filtered with respect to attention
    :rtype NDArrayFp64
    """
    if np.sum(boolean_index) < fps:
        return None

    distraction_tolerance = round(maximum_seconds_distraction * fps)
    minimum_frames_attention = round(minimum_seconds_attention * fps)

    length = len(boolean_index)
    attention_boolean_index = np.zeros(length, dtype=np.bool_)

    i, true_counter, distraction_counter, start = 0, 0, 0, 0
    while i < length:
        if boolean_index[i]:
            if true_counter >= minimum_frames_attention:
                start = i - true_counter  # equivalent to: i - minimum_frames_attention
                true_counter = 0
            else:
                true_counter += 1
        else:
            if start:
                distraction_counter += 1
                if distraction_counter == distraction_tolerance:
                    attention_boolean_index[start : i - distraction_counter] = True
                    i += distraction_counter
                    start, distraction_counter = 0, 0
            else:
                true_counter = 0

        i += 1

    return attention_boolean_index


if ENABLE_NUMBA:
    _filter = njit(cache=True)(_filter)
