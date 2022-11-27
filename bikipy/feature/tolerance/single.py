import numpy as np
from numba import njit
from pydantic import validate_arguments
from pydantic_numpy.dtype import NDArrayBool, NDArrayInt64

from bikipy import runtime_settings
from bikipy.feature.tolerance.common import (
    common_preparation,
    tolerance_filter_warning_wrapper,
)


@validate_arguments
def single_node_tolerance_filter(
    boolean_index: NDArrayBool,
    fps: float,
    minimum_seconds_attention: float = runtime_settings.minimum_seconds_tolerance,
    maximum_seconds_distraction: float = runtime_settings.maximum_seconds_distraction,
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

    minimum_frames_attention, distraction_tolerance, length = common_preparation(
        minimum_seconds_attention, maximum_seconds_distraction, fps, boolean_index
    )

    attention_boolean_index = np.zeros(length, dtype=np.bool_)

    i, true_counter, distraction_counter, start = 0, 0, 0, 0
    while i < length:
        if boolean_index[i]:
            if start:
                # We don't want to use the true counter during a TRUE epoch
                pass
            elif true_counter >= minimum_frames_attention:
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
            elif true_counter > 0:
                true_counter -= 1

        i += 1

    return attention_boolean_index


def arg_single_node_tolerance_filter(
    boolean_index: NDArrayBool,
    fps: float,
    minimum_seconds_attention: float = runtime_settings.minimum_seconds_tolerance,
    maximum_seconds_distraction: float = runtime_settings.maximum_seconds_distraction,
) -> list[NDArrayInt64, ...]:
    """
    Deal with islands of data that need to be aggregated for analysis. These islands
    of data have to be merged arbitrarily.

    A simple merge would make the computation of speed and acceleration wrong.
    """
    if np.sum(boolean_index) < fps:
        return [np.array((x, x, x)) for x in range(0)]

    minimum_frames_attention, distraction_tolerance, length = common_preparation(
        minimum_seconds_attention, maximum_seconds_distraction, fps, boolean_index
    )

    data = []
    i, true_counter, distraction_counter, start = 0, 0, 0, 0
    while i < length:
        if boolean_index[i]:
            if start:
                # We don't want to use the true counter during a TRUE epoch
                pass
            elif true_counter >= minimum_frames_attention:
                start = i - true_counter  # equivalent to: i - minimum_frames_attention
                true_counter = 0
            else:
                true_counter += 1
        else:
            if start:

                distraction_counter += 1
                if distraction_counter == distraction_tolerance:
                    end = i - distraction_counter
                    data.append(np.array((start, end, end - start)))

                    i += distraction_counter
                    start, distraction_counter = 0, 0

            elif true_counter > 0:
                true_counter -= 1

        i += 1

    return data


if not runtime_settings.disable_numba:
    _filter = njit(cache=True)(_filter)

    arg_single_node_tolerance_filter = njit(cache=True)(arg_single_node_tolerance_filter)
