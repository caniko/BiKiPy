import numpy as np
from numba import njit
from pydantic import validate_arguments
from pydantic_numpy.dtype import NDArrayBool

from bikipy import runtime_settings
from bikipy.feature.tolerance.common import (
    common_preparation,
    tolerance_filter_warning_wrapper,
)


@validate_arguments
def plural_node_tolerance_filter(
    *boolean_indices: NDArrayBool,
    fps: float,
    minimum_seconds_attention: float = runtime_settings.minimum_seconds_tolerance,
    maximum_seconds_distraction: float = runtime_settings.maximum_seconds_distraction,
) -> NDArrayBool | None:
    all_true = np.logical_and.reduce(boolean_indices)
    any_true = np.logical_or.reduce(boolean_indices)

    return tolerance_filter_warning_wrapper(
        _filter,
        len(boolean_indices[0]),
        all_true,
        any_true,
        fps,
        minimum_seconds_attention,
        maximum_seconds_distraction,
    )


def _filter(
    all_true: NDArrayBool,
    any_true: NDArrayBool,
    fps: float,
    minimum_seconds_attention: float = runtime_settings.minimum_seconds_tolerance,
    maximum_seconds_distraction: float = runtime_settings.maximum_seconds_distraction,
) -> NDArrayBool | None:
    """
    Combines boolean indices into one boolean index into one. We do this with both an AND and OR filter, yielding two
    datasets; "all_true" and "any_true". We also flip the "any_true" dataset to get "any_".

    1) All nodes must be TRUE for N seconds, defined by minimum_seconds_attention, for TRUE instance to being. We use
    the "all_true" filter here.

    2) After the TRUE event starts we track distraction by observing when "any true" becomes FALSE. When it is FALSE for
    M seconds, defined by maximum_seconds_distraction.

    Event though it should be None, start is set to 0 to support njit mode in numba.

    :param boolean_indices:
    :param fps:
    :param minimum_seconds_attention:
    :param maximum_seconds_distraction:
    :return:
    """
    if np.sum(all_true) < fps:
        return None

    minimum_frames_attention, distraction_tolerance, length = common_preparation(
        minimum_seconds_attention, maximum_seconds_distraction, fps, all_true
    )
    attention_boolean_index = np.zeros(length, dtype=np.bool_)

    i, true_counter, distraction_counter, start = 0, 0, 0, 0
    while i < length:
        if all_true[i]:
            true_counter += 1
            if true_counter >= minimum_frames_attention:
                start = i - true_counter  # equivalent to: i - minimum_frames_attention
                true_counter = 0

                # TRUE instance
                while i < length:
                    if any_true[i]:
                        if distraction_counter > 0:
                            distraction_counter -= 1
                    else:
                        distraction_counter += 1

                        if distraction_counter >= distraction_tolerance:
                            attention_boolean_index[start:i] = True
                            i += 1 + distraction_counter
                            start, distraction_counter = 0, 0
                            break
                    i += 1
                if start:
                    attention_boolean_index[start:i] = True

        elif true_counter > 0:
            true_counter -= 1

        i += 1

    return attention_boolean_index


if not runtime_settings.disable_numba:
    _filter = njit(cache=True)(_filter)
