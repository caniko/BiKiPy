import numpy as np

from bikipy.core.typing import NDArrayBool
from bikipy.feature.tolerance import GENERIC_MINIMUM_SECONDS_ATTENTION, GENERIC_MAXIMUM_SECONDS_DISTRACTION


def plural_node_tolerance_filter(
    *boolean_indices,
    fps: float,
    minimum_seconds_attention: float = GENERIC_MINIMUM_SECONDS_ATTENTION,
    maximum_seconds_distraction: float = GENERIC_MAXIMUM_SECONDS_DISTRACTION,
) -> NDArrayBool:
    """
    Combines boolean indices into one boolean index into one. We do this with both an AND and OR filter, yielding two
    datasets; "all_true" and "any_true". We also flip the "any_true" dataset to get "any_".

    1) All nodes must be TRUE for N seconds, defined by minimum_seconds_attention, for TRUE instance to being. We use
    the "all_true" filter here.

    2) After the TRUE event starts we track distraction by observing when "any true" becomes FALSE. When it is FALSE for
    M seconds, defined by maximum_seconds_distraction.


    :param boolean_indices:
    :param fps:
    :param minimum_seconds_attention:
    :param maximum_seconds_distraction:
    :return:
    """
    all_true = np.logical_and.reduce(boolean_indices)
    any_true = np.logical_or.reduce(boolean_indices)

    if np.sum(all_true) < fps:
        return np.zeros_like(all_true, dtype=bool)

    distraction_tolerance = round(maximum_seconds_distraction * fps)
    minimum_frames_attention = round(minimum_seconds_attention * fps)

    length = all_true.shape[0]
    attention_boolean_index = np.zeros(length, dtype=bool)

    length = len(all_true)
    i, true_counter, distraction_counter, start = 0, 0, 0, None
    while i < length:
        if all_true[i]:
            true_counter += 1
            if true_counter >= minimum_frames_attention:
                start = i - true_counter  # equivalent to: i - minimum_frames_attention
                true_counter = 0

                # TRUE instance
                while i + distraction_counter < length:
                    if any_true[i + distraction_counter]:
                        i += 1 + distraction_counter
                        distraction_counter = 0
                    else:
                        distraction_counter += 1

                        if distraction_counter >= distraction_tolerance:
                            attention_boolean_index[start:i] = True
                            i += 1 + distraction_counter
                            start, distraction_counter = None, 0
                            break
                #
                if start:
                    attention_boolean_index[start:i] = True
                    break
        else:
            true_counter = 0

        i += 1

    return attention_boolean_index
