from logging import getLogger
from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sb
from pydantic import validate_arguments
from pydantic_numpy.dtype import NDArrayBool, NDArrayFp64

from bikipy import runtime_settings
from bikipy.core.video import VideoMetadata, inspect_video_is_none_during_inspection
from bikipy.perimeter.base import SinglePerimeter
from bikipy.utils.image import axis_frame_imshow
from bikipy.utils.plot import BOTTOM_LEGEND_KWARGS
from bikipy.utils.plot.color import make_color_map

logger = getLogger(__name__)


@validate_arguments
def proximity_filter(
    perimeter: SinglePerimeter,
    perimeter_border_normal_pixels: float | NDArrayFp64,
    inside_perimeter_border: NDArrayFp64,
    outside_perimeter_border: Optional[NDArrayFp64] = None,
    outside_perimeter: Optional[NDArrayBool] = None,
    inspect_video: Optional[VideoMetadata] = None,
    inspect: bool = False,
    inspect_pixels: bool = False,
    manual_ax: Any = None,
) -> NDArrayBool:
    perimeter_border = perimeter.expand(perimeter_border_normal_pixels)
    inside_perimeter_border = perimeter_border.confined_coordinate(coordinates=inside_perimeter_border)
    result = inside_perimeter_border

    if perimeter.impenetrable:
        outside_impenetrable_bi = ~perimeter.confined_coordinate(coordinates=inside_perimeter_border)
        result = result & outside_impenetrable_bi

    if outside_perimeter_border is not None:
        outside_perimeter_border_bi = ~perimeter_border.confined_coordinate(outside_perimeter_border)
        result = result & outside_perimeter_border_bi

    if outside_perimeter is not None:
        outside_perimeter_bi = ~perimeter.confined_coordinate(outside_perimeter_border)
        result = result & outside_perimeter_bi

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

        ax.set_title("Proximity detection", fontsize=inspect_video.upscaled_video.plotting_title_font_size)

        perimeter.plot(ax=ax, inspect_pixels=inspect_pixels)
        perimeter_border.plot(ax=ax, inspect_pixels=inspect_pixels)

        color_count = 1
        if perimeter.impenetrable:
            color_count += 1
        if outside_perimeter_border is not None:
            color_count += 1
        if outside_perimeter is not None:
            color_count += 1
        color_map_iter = iter(make_color_map(color_count))

        ax.scatter(
            *inside_perimeter_border_plot_scaled[result].T,
            marker="x",
            alpha=runtime_settings.matplotlib_scatter_alpha,
            label="Valid",
            color=next(color_map_iter),
        )

        not_result = ~result
        if perimeter.impenetrable:
            ax.scatter(
                *inside_perimeter_border_plot_scaled[outside_impenetrable_bi & not_result].T,
                marker="x",
                alpha=runtime_settings.matplotlib_scatter_alpha,
                label="Outside impenetrable valid; other invalid",
                color=next(color_map_iter),
            )
        if outside_perimeter_border is not None:
            ax.scatter(
                *inside_perimeter_border_plot_scaled[outside_perimeter_border_bi & not_result].T,
                marker="x",
                alpha=runtime_settings.matplotlib_scatter_alpha,
                label="Outside border valid; other invalid",
                color=next(color_map_iter),
            )
        if outside_perimeter is not None:
            ax.scatter(
                *inside_perimeter_border_plot_scaled[outside_perimeter_bi & not_result].T,
                marker="x",
                alpha=runtime_settings.matplotlib_scatter_alpha,
                label="Outside perimeter valid; other invalid",
                color=next(color_map_iter),
            )

        ax.legend(**BOTTOM_LEGEND_KWARGS, fontsize=inspect_video.upscaled_video.plotting_default_font_size)

        if not manual_ax:
            plt.tight_layout()
            plt.show()

    return result
