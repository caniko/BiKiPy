from collections import Sequence
from logging import getLogger

import numpy as np


logger = getLogger(__name__)


def attention_per_frame(
    boolean_index: Sequence[bool],
    fps: float,
    minimum_seconds_attention: float = 0.2,
) -> np.ndarray:
    """
    Filters boolean_index with respect to attention. The filter tolerates distraction, and requires
    minimum_seconds_attention to be fulfilled before accepting the sequence as attention.

    :param boolean_index:
    :param fps: Frames per second (fps) of the recording used to generate the data in boolean_index
    :param minimum_seconds_attention: Minimum number of seconds that the sequence has to be True
    for it to be defined as an attention sequence. Sequences that fall short of this lowpass filter
    will be converted to False.
    :type boolean_index: np.ndarray
    :type fps: float
    :type minimum_seconds_attention: float
    :return: Boolean index filtered with respect to attention
    :rtype np.ndarray
    """
    boolean_index = np.asarray(boolean_index)

    fps = float(fps)
    minimum_seconds_attention = float(minimum_seconds_attention)

    distraction_tolerance = round(fps / 2.0)
    minimum_time_valid_observation = round(minimum_seconds_attention * fps)

    length = boolean_index.shape[0]
    attention_boolean_index = np.zeros(length, dtype=bool)

    first_valid_index = None
    i, valid_frames_within_border, true_counter, consecutive_false = 0, 0, 0, 0
    while True:
        if boolean_index[i]:
            true_counter += 1

            if consecutive_false:
                true_counter += consecutive_false - 1  # make up for the previous += 1
                consecutive_false = 0

            if true_counter == minimum_time_valid_observation:
                # The first valid index is the index of the first True, i.e. when true_counter was 1
                first_valid_index = i - minimum_time_valid_observation + 1

        else:
            if first_valid_index is not None:
                if consecutive_false <= distraction_tolerance:
                    consecutive_false += 1
                else:
                    attention_boolean_index[first_valid_index : i + 1] = True
                    valid_frames_within_border += true_counter

                    consecutive_false, true_counter = 0, 0
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
        logger.info(f"Subject didn't observe the nort object")

        assert not np.any(attention_boolean_index)
        return attention_boolean_index

    assert (
        np.any(attention_boolean_index)
        and np.sum(attention_boolean_index) >= minimum_time_valid_observation
    ), (
        f"True: {np.sum(attention_boolean_index)}; fps: {fps}; "
        f"Minimum observation frames: {minimum_time_valid_observation}"
    )

    return attention_boolean_index
