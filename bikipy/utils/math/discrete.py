import numpy as np
from numba import njit
from pydantic_numpy import NDArrayBool

from bikipy import runtime_settings
from bikipy.feature.tolerance.common import common_preparation

TruthIslandMetadata = list[tuple[int, int, int]]


def boolean_index_truth_sequence_start_end(
    boolean_index: NDArrayBool, with_length: bool = False
) -> TruthIslandMetadata | list[tuple[int, int]]:
    result = []

    array_length = len(boolean_index)
    idx = 0

    while idx < array_length:
        if boolean_index[idx]:
            start = idx
            if start != 0:
                start -= 1

            while boolean_index[idx] and idx < array_length:
                idx += 1

            result.append((start, idx, idx - start) if with_length else (start, idx))

        idx += 1

    return result


def tolerance_modeled_boolean_index_truth_sequence_start_end_length(
    boolean_index: NDArrayBool,
    fps: float,
    minimum_seconds_attention: float = runtime_settings.minimum_seconds_tolerance,
    maximum_seconds_distraction: float = runtime_settings.maximum_seconds_distraction,
) -> tuple[TruthIslandMetadata, np.ndarray[bool, bool]]:
    """
    Deal with islands of data that need to be aggregated for analysis. These islands
    of data have to be merged arbitrarily.

    A simple merge would make the computation of speed and acceleration wrong.
    """
    if np.sum(boolean_index) < fps:
        return [], boolean_index

    minimum_frames_attention, distraction_tolerance, length = common_preparation(
        minimum_seconds_attention, maximum_seconds_distraction, fps, boolean_index
    )

    new_boolean_index = np.zeros_like(boolean_index, dtype=bool)
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

            if distraction_counter:
                distraction_counter -= 1

        else:
            if start:
                distraction_counter += 1
                if distraction_counter == distraction_tolerance:
                    end = i - distraction_counter
                    new_boolean_index[start:end] = True
                    data.append((start, end, end - start))

                    i += distraction_counter
                    start, distraction_counter = 0, 0

            elif true_counter > 0:
                true_counter -= 1

        i += 1

    return data, new_boolean_index


if not runtime_settings.disable_numba:
    boolean_index_truth_sequence_start_end = njit(cache=True)(boolean_index_truth_sequence_start_end)

    tolerance_modeled_boolean_index_truth_sequence_start_end_length = njit(cache=True)(
        tolerance_modeled_boolean_index_truth_sequence_start_end_length
    )
