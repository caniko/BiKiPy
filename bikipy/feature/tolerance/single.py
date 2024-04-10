import numpy as np
from numba import njit
from pydantic import validate_call
from pydantic_numpy.typing import Np1DArrayBool

from bikipy import runtime_settings
from bikipy.feature.tolerance.common import (
    common_preparation,
    tolerance_model_warning_wrapper,
)


@validate_call
def single_node_tolerance_model(
    boolean_index: Np1DArrayBool,
    fps: float,
    minimum_seconds_attention: float = runtime_settings.minimum_seconds_tolerance,
    maximum_seconds_distraction: float = runtime_settings.maximum_seconds_distraction,
) -> Np1DArrayBool | None:
    return tolerance_model_warning_wrapper(
        _filter, len(boolean_index), boolean_index, fps, minimum_seconds_attention, maximum_seconds_distraction
    )


def _filter(
    boolean_index: Np1DArrayBool,
    fps: float,
    minimum_seconds_attention: float,
    maximum_seconds_distraction: float,
) -> Np1DArrayBool | None:
    """
    MinFA and MaxFD are used to tolerance model the provided binary sequence as follows:

    #. :code:`True` must persist for MinFA elements for a tolerated sequence to *start*, and we set the
                beginning of the sequence to the index of the first :code:`True` value in the sequence.

    .. note::
       The entirety of the tolerated sequence will be set to :code:`True`

    #. Every :code:`False` will accumulate to a distraction counter till the counter is equal to MaxFD.

    .. note::
       When :code:`True`, and the distraction counter is more than 0, decrement by 1.

    #. The tolerance sequence is terminated at the index before the final :code:`False` element.


    :param boolean_index:
    :param fps: Frames per second (fps) of the recording used to generate the data in boolean_index
    :param minimum_seconds_attention: Minimum number of seconds that the sequence has to be True
    for it to be defined as an attention sequence. Filtered sequences will be converted to False.
    :param maximum_seconds_distraction:
    :type boolean_index: NpNDArrayFp64
    :type fps: float
    :type minimum_seconds_attention: float
    :return: Boolean index filtered with respect to attention
    :rtype NpNDArrayFp64
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

            if distraction_counter:
                distraction_counter -= 1

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


if not runtime_settings.disable_numba:
    _filter = njit(cache=True)(_filter)
