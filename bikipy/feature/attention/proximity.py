from logging import getLogger
from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sb
from pydantic import validate_arguments
from pydantic_numpy.dtype import NDArrayBool, NDArrayFp64

from bikipy import runtime_settings
from bikipy.core.video import (
    VideoMetadata,
    inspect_video_is_none_during_inspection,
)
from bikipy.perimeter.base import SinglePerimeter
from bikipy.utils.image import axis_frame_imshow
from bikipy.utils.plotting import BOTTOM_LEGEND_KWARGS, make_color_map

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
                axis_frame_imshow(ax, perimeter.inspect_image)
        else:
            ax = manual_ax

        inside_perimeter_border_plot_scaled = inspect_video.prepare_coordinates_for_plotting(
            inside_perimeter_border, inspect_pixels
        )

        if inspect_pixels:
            inspect_video.upscaled_video.ax_ticks_metric_to_pixel(ax)

        ax.set_title("Proximity filter", fontsize=inspect_video.upscaled_video.plotting_title_font_size)

        perimeter.plot(
            ax=ax,
            inspect_pixels=inspect_pixels,
        )
        perimeter_border.plot(
            ax=ax,
            inspect_pixels=inspect_pixels,
        )

        color_map = make_color_map(2 if perimeter.impenetrable else 3)

        ax.scatter(
            *inside_perimeter_border_plot_scaled[result].T,
            marker="x",
            alpha=runtime_settings.matplotlib_scatter_alpha,
            label="Valid",
            color=color_map[0],
        )

        not_result = ~result
        if perimeter.impenetrable:
            ax.scatter(
                *inside_perimeter_border_plot_scaled[not_result].T,
                marker="x",
                alpha=runtime_settings.matplotlib_scatter_alpha,
                label="Invalid",
                color=color_map[1],
            )
        else:
            ax.scatter(
                *inside_perimeter_border_plot_scaled[inside_perimeter_border_boolean_index & not_result].T,
                marker="x",
                alpha=runtime_settings.matplotlib_scatter_alpha,
                label="Nose valid, invalid torso",
                color=color_map[1],
            )
            ax.scatter(
                *inside_perimeter_border_plot_scaled[outside_perimeter_boolean_index & not_result].T,
                marker="x",
                alpha=runtime_settings.matplotlib_scatter_alpha,
                label="Torso valid, invalid nose",
                color=color_map[2],
            )

        ax.legend(**BOTTOM_LEGEND_KWARGS, fontsize=inspect_video.upscaled_video.plotting_default_font_size)

        if not manual_ax:
            plt.tight_layout()
            plt.show()

    return result
