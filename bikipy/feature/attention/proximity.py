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
from bikipy.perimeter.base import SinglePerimeter

logger = getLogger(__name__)


@validate_arguments
def proximity_filter(
    perimeter: SinglePerimeter,
    inside_perimeter_border: NDArrayFp64,
    outside_perimeter: NDArrayFp64,
    perimeter_border_normal_pixels: float | NDArrayFp64,
    inspect_video: Optional[VideoMetadata] = None,
    inspect: bool = False,
    inspect_pixels: bool = False,
    manual_ax: Any = None,
) -> NDArrayBool:
    """
    Filter with respect to proximity rules. (1) The inside_perimeter_border has to be in front of perimeter, but inside the perimeter;
    (2) the outside_perimeter is outside the perimeter.

    :param perimeter:
    :param inside_perimeter_border: Cartesian coordinates of the inside_perimeter_border
    :param outside_perimeter: Cartesian coordinates of the center of mass
    :param perimeter_border_normal_pixels: The magnitude of the normal between the perimeter and the perimeter in pixels
    :param inspect: If True, generate and view an analytics of the resulting filter
    :param manual_ax: matplotlib Axes that the inspection plots will (optionally) be saved in
    :type perimeter: SinglePerimeter
    :type inside_perimeter_border: NDArrayFp64
    :type outside_perimeter: NDArrayFp64
    :type perimeter_border_normal_pixels: float | NDArrayFp64
    :type inspect: bool
    :type manual_ax: Any
    :return:
    :rtype: NDArrayFp64
    """
    # Remove inside_perimeter_border points that aren't inside the perimeter
    perimeter_border = perimeter.expand(perimeter_border_normal_pixels)
    inside_perimeter_border_boolean_index = perimeter_border.confined_coordinate_boolean_index(
        coordinates=inside_perimeter_border
    )

    if perimeter.impenetrable:
        result = inside_perimeter_border_boolean_index
    else:
        outside_perimeter_boolean_index = ~perimeter.confined_coordinate_boolean_index(outside_perimeter)
        result = inside_perimeter_border_boolean_index & outside_perimeter_boolean_index

    if manual_ax is not None or inspect:
        inspect_video_is_none_during_inspection(inspect_video)

        if manual_ax is None:
            sb.set_theme(style="darkgrid")
            fig, ax = plt.subplots(dpi=300)
            if np.any(perimeter.inspect_image):
                ax.imshow(perimeter.inspect_image)
        else:
            ax = manual_ax

        if inspect_pixels:
            inside_perimeter_border = convert_meters_to_pixels(inside_perimeter_border, inspect_video)

        ax.set_title("Proximity filter")

        perimeter.plot(
            ax=ax,
            inspect_pixels=inspect_pixels,
        )
        perimeter_border.plot(
            ax=ax,
            inspect_pixels=inspect_pixels,
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

        # ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.025), fancybox=True, ncol=3)

        if not manual_ax:
            plt.tight_layout()
            plt.show()

    return result
