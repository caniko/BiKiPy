from logging import getLogger

import numpy as np
from pydantic import validate_arguments

from bikipy.core.typing import NDArrayBool
from bikipy.feature.tolerance import GENERIC_MAXIMUM_SECONDS_DISTRACTION, GENERIC_MINIMUM_SECONDS_ATTENTION

logger = getLogger(__name__)


@validate_arguments
def single_node_tolerance_filter(
    boolean_index: NDArrayBool,
    fps: float,
    minimum_seconds_attention: float = GENERIC_MINIMUM_SECONDS_ATTENTION,
    maximum_seconds_distraction: float = GENERIC_MAXIMUM_SECONDS_DISTRACTION,
) -> NDArrayBool:
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
        return np.zeros_like(boolean_index, dtype=bool)

    distraction_tolerance = round(maximum_seconds_distraction * fps)
    minimum_frames_attention = round(minimum_seconds_attention * fps)

    length = boolean_index.shape[0]
    attention_boolean_index = np.zeros(length, dtype=bool)

    first_valid_index = None
    i, valid_frames_within_border, true_counter, consecutive_false, frames_after_distraction = 0, 0, 0, 0, 0
    while True:
        if boolean_index[i]:
            if consecutive_false:
                if frames_after_distraction == minimum_frames_attention:
                    true_counter += consecutive_false + minimum_frames_attention
                    consecutive_false, frames_after_distraction = 0, 0
                else:
                    frames_after_distraction += 1
            else:
                true_counter += 1

            if true_counter == minimum_frames_attention:
                # The first valid index is the index of the first True, i.e. when true_counter was 1
                first_valid_index = i - minimum_frames_attention + 1

        else:
            if first_valid_index is not None:
                if consecutive_false + frames_after_distraction <= distraction_tolerance:
                    consecutive_false += 1
                else:
                    attention_boolean_index[first_valid_index : i + 1] = True
                    valid_frames_within_border += true_counter

                    consecutive_false, frames_after_distraction, true_counter = 0, 0, 0
                    first_valid_index = None

            else:
                true_counter = 0

        i += 1

        if i == length:
            if first_valid_index is not None:
                attention_boolean_index[first_valid_index:] = True
                valid_frames_within_border += true_counter
            break

    if valid_frames_within_border == 0:
        logger.info(f"Subject didn't observe the polygonal perimeter")

        assert not np.any(attention_boolean_index)
        return attention_boolean_index

    assert np.any(attention_boolean_index) and np.sum(attention_boolean_index) >= minimum_frames_attention, (
        f"True: {np.sum(attention_boolean_index)}; fps: {fps}; "
        f"Minimum observation frames: {minimum_frames_attention}"
    )

    return attention_boolean_index
