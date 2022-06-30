from logging import getLogger
from typing import Any, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sb
from pydantic import validate_arguments

from bikipy import MATPLOTLIB_SCATTER_ALPHA
from bikipy.core.typing import NDArrayBool, NDArrayFp64
from bikipy.core.video import (
    VideoMetadata,
    convert_meters_to_pixels,
    inspect_video_is_none_during_inspection,
)
from bikipy.perimeter.base import AnyPerimeter

logger = getLogger(__name__)


@validate_arguments
def proximity_filter(
    perimeter: AnyPerimeter,
    inside_perimeter_border: NDArrayFp64,
    outside_perimeter: NDArrayFp64,
    perimeter_border_normal_meters: float | NDArrayFp64,
    inspect_video: Optional[VideoMetadata] = None,
    inspect: bool = False,
    inspect_pixels: bool = False,
    inspection_ax: Any = None,
) -> NDArrayBool:
    """
    Filter with respect to proximity rules. (1) The inside_perimeter_border has to be in front of perimeter, but inside the perimeter;
    (2) the outside_perimeter is outside the perimeter.

    :param perimeter:
    :param inside_perimeter_border: Cartesian coordinates of the inside_perimeter_border
    :param outside_perimeter: Cartesian coordinates of the center of mass
    :param perimeter_border_normal_meters: The magnitude of the normal between the perimeter and the perimeter in meters
    :param inspect: If True, generate and view an analytics of the resulting filter
    :param inspection_ax: matplotlib Axes that the inspection plots will (optionally) be saved in
    :type perimeter: AnyPerimeter
    :type inside_perimeter_border: NDArrayFp64
    :type outside_perimeter: NDArrayFp64
    :type perimeter_border_normal_meters: float | NDArrayFp64
    :type inspect: bool
    :type inspection_ax: Any
    :return:
    :rtype: NDArrayFp64
    """
    # Remove inside_perimeter_border points that aren't inside the perimeter
    perimeter_border = perimeter.expand(perimeter_border_normal_meters)

    if perimeter.impenetrable:
        inside_perimeter_border_boolean_index = perimeter_border.coordinate_confinement_boolean_index(
            coordinates=inside_perimeter_border
        )
        result = inside_perimeter_border_boolean_index
    else:
        inside_perimeter_border_boolean_index = perimeter_border.coordinate_confinement_boolean_index(
            coordinates=inside_perimeter_border
        )
        outside_perimeter_boolean_index = ~perimeter.coordinate_confinement_boolean_index(outside_perimeter)

        result = inside_perimeter_border_boolean_index & outside_perimeter_boolean_index

    if inspection_ax is not None or inspect:
        inspect_video_is_none_during_inspection(inspect_video)

        if inspection_ax is None:
            sb.set_theme(style="darkgrid")
            fig, ax = plt.subplots(dpi=300)
            if np.any(perimeter.inspect_image):
                ax.imshow(perimeter.inspect_image)
        else:
            ax = inspection_ax

        if inspect_pixels:
            inside_perimeter_border = convert_meters_to_pixels(inside_perimeter_border, inspect_video)

        ax.set_title("Proximity filter")

        perimeter.plot(
            ax=ax,
            inspect_pixels=inspect_pixels,
            perimeter_border_normal_pixels=perimeter_border_normal_meters * inspect_video.pixels_per_meter,
        )
        perimeter_border.plot(
            ax=ax,
            inspect_pixels=inspect_pixels,
            perimeter_border_normal_pixels=perimeter_border_normal_meters * inspect_video.pixels_per_meter,
        )

        ax.scatter(*inside_perimeter_border[result].T, marker=",", alpha=MATPLOTLIB_SCATTER_ALPHA, label="Valid")

        not_result = ~result
        if perimeter.impenetrable:
            ax.scatter(
                *inside_perimeter_border[not_result].T,
                marker=",",
                alpha=MATPLOTLIB_SCATTER_ALPHA,
                label="Invalid",
            )
        else:
            ax.scatter(
                *inside_perimeter_border[inside_perimeter_border_boolean_index & not_result].T,
                marker=",",
                alpha=MATPLOTLIB_SCATTER_ALPHA,
                label="Nose valid, invalid outside_perimeter",
            )
            ax.scatter(
                *inside_perimeter_border[outside_perimeter_boolean_index & not_result].T,
                marker=",",
                alpha=MATPLOTLIB_SCATTER_ALPHA,
                label="Center of mass valid, invalid inside_perimeter_border",
            )

        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.025), fancybox=True, ncol=3)

        if not inspection_ax:
            plt.tight_layout()
            plt.show()

    return result


@validate_arguments
def tolerance_filter(
    boolean_index: NDArrayBool,
    fps: float,
    minimum_seconds_attention: float,
    maximum_seconds_distraction: float = 0.5,
) -> NDArrayBool:
    """
    Filters boolean_index with respect to attention. The filter tolerates distraction, and requires
    minimum_seconds_attention to be fulfilled before accepting the sequence as attention.

    1. Including an attention event requires attention time to be greater than minimum_seconds_attention
    2. During an attention event, the subject may be distracted for maximum_seconds_distraction seconds.
       This triggers a sub event:
        a) The subject has to be attentive for minimum_seconds_attention to merge the gap between the new attention
           with the previous.
        b) The events will remain if the distraction time surpasses the maximum_seconds_distraction. Note that the new
           attention might be removed if it is shorter than minimum_seconds_attention

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
