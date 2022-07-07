import numpy as np
from numba import njit
from pydantic import validate_arguments

from bikipy import ENABLE_NUMBA
from bikipy.core.typing import NDArrayBool
from bikipy.feature.tolerance import GENERIC_MINIMUM_SECONDS_ATTENTION, GENERIC_MAXIMUM_SECONDS_DISTRACTION
from bikipy.feature.tolerance.common import tolerance_filter_warning_wrapper


@validate_arguments
def plural_node_tolerance_filter(
    *boolean_indices,
    fps: float,
    minimum_seconds_attention: float = GENERIC_MINIMUM_SECONDS_ATTENTION,
    maximum_seconds_distraction: float = GENERIC_MAXIMUM_SECONDS_DISTRACTION,
) -> NDArrayBool | None:
    return tolerance_filter_warning_wrapper(
        _filter, len(boolean_indices[0]), *boolean_indices, fps, minimum_seconds_attention, maximum_seconds_distraction
    )


def _filter(
    *boolean_indices,
    fps: float,
    minimum_seconds_attention: float = GENERIC_MINIMUM_SECONDS_ATTENTION,
    maximum_seconds_distraction: float = GENERIC_MAXIMUM_SECONDS_DISTRACTION,
) -> NDArrayBool | None:
    """
    Combines boolean indices into one boolean index into one. We do this with both an AND and OR filter, yielding two
    datasets; "all_true" and "any_true". We also flip the "any_true" dataset to get "any_".

    1) All nodes must be TRUE for N seconds, defined by minimum_seconds_attention, for TRUE instance to being. We use
    the "all_true" filter here.

    2) After the TRUE event starts we track distraction by observing when "any true" becomes FALSE. When it is FALSE for
    M seconds, defined by maximum_seconds_distraction.

    start is set to 0 when in fact it should be None to support njit mode in numba.

    :param boolean_indices:
    :param fps:
    :param minimum_seconds_attention:
    :param maximum_seconds_distraction:
    :return:
    """
    all_true = np.logical_and.reduce(boolean_indices)
    any_true = np.logical_or.reduce(boolean_indices)

    if np.sum(all_true) < fps:
        return None

    distraction_tolerance = round(maximum_seconds_distraction * fps)
    minimum_frames_attention = round(minimum_seconds_attention * fps)

    length = len(all_true)
    attention_boolean_index = np.zeros(length, dtype=np.bool_)

    i, true_counter, distraction_counter, start = 0, 0, 0, 0
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
                            start, distraction_counter = 0, 0
                            break
                if start:
                    attention_boolean_index[start:i] = True
                    break
        else:
            true_counter = 0

        i += 1

    return attention_boolean_index


if ENABLE_NUMBA:
    _filter = njit(cache=True)(_filter)
