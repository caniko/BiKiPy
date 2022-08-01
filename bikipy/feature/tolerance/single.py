import numpy as np
from numba import njit
from pydantic import validate_arguments

from bikipy import ENABLE_NUMBA
from bikipy.core.typing import NDArrayBool
from bikipy.feature.tolerance import GENERIC_MINIMUM_SECONDS_ATTENTION, GENERIC_MAXIMUM_SECONDS_DISTRACTION
from bikipy.feature.tolerance.common import tolerance_filter_warning_wrapper, common_preparation


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

    distraction_tolerance, minimum_frames_attention, length = common_preparation(
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


def tolerated_islands(
    boolean_index: NDArrayBool,
    fps: float,
    minimum_seconds_attention: float,
    maximum_seconds_distraction: float,
):
    """
    Deal with islands of data that need to be aggregated for analysis. These islands
    of data have to be merged arbitrarily.

    A simple merge would make the computation of speed and acceleration wrong.
    """
    if np.sum(boolean_index) < fps:
        return []

    distraction_tolerance, minimum_frames_attention, length = common_preparation(
        minimum_seconds_attention, maximum_seconds_distraction, fps, boolean_index
    )

    last_index = length - 1
    finder_result = find_index_start_n_end()

    if not finder_result:
        return []
    i, start, _end = finder_result

    data = []
    while i < length:
        potential_end = indexes[i]
        next_step_from_previous_end = indexes[i - 1] + 1
        if potential_end == next_step_from_previous_end:
            pass
        elif potential_end > next_step_from_previous_end:
            jump_length = potential_end - next_step_from_previous_end
            if jump_length <= third_of_a_second:
                end = potential_end

                # Look ahead before committing to end index
                if i != last_index and end - indexes[i + 1] < third_of_a_second:
                    i += 1
                    continue

            else:
                end = indexes[i - 1]

            if end - start > minimum_frames:
                data.append((start, end))

                if i == last_index:
                    break

                finder_result = find_index_start_n_end()

                if not finder_result:
                    break
                i, start, _end = finder_result

                continue
        else:  # potential_end < next_step_from_previous_end
            msg = "potential_end < next_step_from_end cannot be true in a sorted index"
            raise RuntimeError(msg)

        i += 1

    return data


def find_index_start_n_end(starting_index: int = 0):
    new_start, new_end = indexes[starting_index], indexes[starting_index := starting_index + 1]
    while new_end - new_start > fps:
        new_start, new_end = indexes[starting_index], indexes[starting_index := starting_index + 1]
        if starting_index < length:
            return False
    return starting_index + 1, new_start, new_end


if ENABLE_NUMBA:
    _filter = njit(cache=True)(_filter)
