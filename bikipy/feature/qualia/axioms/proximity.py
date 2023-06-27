from functools import cached_property
from logging import getLogger
from typing import Optional

import numpy as np
from matplotlib.axes import Axes
from pydantic import validate_arguments
from pydantic_numpy.dtype import NDArrayFp64

from bikipy import runtime_settings
from bikipy.core.compute import AbstractComputeBooleanIndex, T
from bikipy.core.video import VideoMetadata
from bikipy.perimeter.base import SinglePerimeter
from bikipy.utils.plot.color import make_color_map

logger = getLogger(__name__)


def _update_result_array(
    result: np.ndarray[bool, bool] | None, new_array: np.ndarray[bool, bool], all_or_none: bool
) -> np.ndarray[bool, bool]:
    if result is None:
        return new_array

    return result & new_array if all_or_none else result | new_array


class ComputeProximity(AbstractComputeBooleanIndex):
    perimeter: SinglePerimeter
    maximum_distance: float | NDArrayFp64

    inside_perimeter: Optional[NDArrayFp64]
    outside_perimeter: Optional[NDArrayFp64]
    inside_perimeter_border: Optional[NDArrayFp64]
    outside_perimeter_border: Optional[NDArrayFp64]

    heuristic_data_sources = (
        "inside_perimeter",
        "outside_perimeter",
        "inside_perimeter_border",
        "outside_perimeter_border",
    )

    @validate_arguments(config={"arbitrary_types_allowed": True})
    def plot(self, ax: Axes, video: Optional[VideoMetadata] = None, inspect_pixels: bool = False) -> None:
        assert self.result is not None

        inside_perimeter_border_plot_scaled = self.inside_perimeter_border

        if video:
            inside_perimeter_border_plot_scaled = video.prepare_coordinates_for_plotting(
                inside_perimeter_border_plot_scaled, inspect_pixels
            )
            if inspect_pixels:
                video.upscaled_video.ax_ticks_metric_to_pixel(ax)

        self.perimeter.plot(ax=ax, inspect_pixels=inspect_pixels)
        self.perimeter_border.plot(ax=ax, inspect_pixels=inspect_pixels)

        color_count = 1
        if self.outside_perimeter_border is not None:
            color_count += 1
        if self.outside_perimeter is not None:
            color_count += 1
        color_map_iter = iter(make_color_map(color_count))

        ax.scatter(
            *inside_perimeter_border_plot_scaled[self.result].T,
            marker="x",
            alpha=runtime_settings.matplotlib_scatter_alpha,
            label="Valid",
            color=next(color_map_iter),
        )

        not_result = ~self.result
        if self.outside_perimeter_border is not None:
            ax.scatter(
                *inside_perimeter_border_plot_scaled[self.outside_perimeter_border_bi & not_result].T,
                marker="x",
                alpha=runtime_settings.matplotlib_scatter_alpha,
                label="Only outside border",
                color=next(color_map_iter),
            )
        if self.outside_perimeter is not None:
            ax.scatter(
                *inside_perimeter_border_plot_scaled[self.outside_perimeter_bi & not_result].T,
                marker="x",
                alpha=runtime_settings.matplotlib_scatter_alpha,
                label="Only outside perimeter",
                color=next(color_map_iter),
            )

        self.plot_finalization(ax, video)

    @cached_property
    def result(self) -> T:
        result = None

        if self.inside_perimeter is not None:
            result = _update_result_array(
                result, self.perimeter.compute_confinement_boolean_index(self.inside_perimeter), False
            )
        if self.outside_perimeter is not None:
            result = _update_result_array(
                result, ~self.perimeter.compute_confinement_boolean_index(self.outside_perimeter), True
            )
        if self.inside_perimeter_border is not None:
            result = _update_result_array(
                result, self.perimeter_border.compute_confinement_boolean_index(self.inside_perimeter_border), False
            )
        if self.outside_perimeter_border is not None:
            result = _update_result_array(
                result, ~self.perimeter_border.compute_confinement_boolean_index(self.outside_perimeter_border), True
            )

        assert isinstance(result, np.ndarray)

        return result

    @cached_property
    def perimeter_border(self) -> SinglePerimeter:
        return self.perimeter.expand(self.maximum_distance)
