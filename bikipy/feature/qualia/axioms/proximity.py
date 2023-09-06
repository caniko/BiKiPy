from functools import cached_property
from logging import getLogger
from typing import Optional

import numpy as np
from matplotlib.axes import Axes
from pydantic import Field, computed_field, validate_call
from pydantic_numpy import NpNDArrayBool
from pydantic_numpy.typing import NpNDArrayFp64

from bikipy import runtime_settings
from bikipy.core.compute import AbstractComputePerimeterBooleanIndex, T
from bikipy.core.video import VideoMetadata
from bikipy.perimeter.base import BasePerimeter, BaseSinglePerimeter
from bikipy.utils.plot.color import make_color_map

logger = getLogger(__name__)


def _update_result_array(result: NpNDArrayBool | None, new_array: NpNDArrayBool, all_or_none: bool) -> NpNDArrayBool:
    if result is None:
        return new_array

    return result & new_array if all_or_none else result | new_array


class ComputeProximity(AbstractComputePerimeterBooleanIndex):
    maximum_distance: float | NpNDArrayFp64

    inside_perimeter: Optional[NpNDArrayFp64] = None
    outside_perimeter: Optional[NpNDArrayFp64] = None
    inside_perimeter_border: Optional[NpNDArrayFp64] = None
    outside_perimeter_border: Optional[NpNDArrayFp64] = None

    heuristic_data_sources = (
        "inside_perimeter",
        "outside_perimeter",
        "inside_perimeter_border",
        "outside_perimeter_border",
    )

    @computed_field  # type: ignore[misc]
    @cached_property
    def perimeter_border(self) -> BaseSinglePerimeter:
        return self.perimeter.expand(self.maximum_distance)

    my_perimeter_to_boolean_index: dict[BasePerimeter, NpNDArrayBool] = Field(default_factory=dict)

    @computed_field  # type: ignore[misc]
    @cached_property
    def result(self) -> T:
        """
        Collecting data for my_perimeter_to_boolean_index along the way.

        TODO: post_init method
        :return:
        """
        result = None

        perimeter_boolean_index = None
        if self.inside_perimeter is not None:
            perimeter_boolean_index = self.perimeter.compute_confinement_boolean_index(self.inside_perimeter)
            result = _update_result_array(result, perimeter_boolean_index, False)

        if self.outside_perimeter is not None:
            outside_perimeter_boolean_index = ~self.perimeter.compute_confinement_boolean_index(self.outside_perimeter)
            if self.inside_perimeter is not None:
                perimeter_boolean_index = perimeter_boolean_index & outside_perimeter_boolean_index
            result = _update_result_array(result, outside_perimeter_boolean_index, True)

        perimeter_border_boolean_index = None
        if self.inside_perimeter_border is not None:
            perimeter_border_boolean_index = self.perimeter_border.compute_confinement_boolean_index(
                self.inside_perimeter_border
            )
            result = _update_result_array(result, perimeter_border_boolean_index, False)

        if self.outside_perimeter_border is not None:
            outside_perimeter_border_boolean_index = ~self.perimeter_border.compute_confinement_boolean_index(
                self.outside_perimeter_border
            )
            if self.inside_perimeter_border is not None:
                perimeter_border_boolean_index = perimeter_border_boolean_index & outside_perimeter_border_boolean_index
            result = _update_result_array(result, outside_perimeter_border_boolean_index, True)

        assert isinstance(result, np.ndarray)

        if perimeter_boolean_index is not None:
            self.my_perimeter_to_boolean_index[self.perimeter] = perimeter_boolean_index
        if perimeter_border_boolean_index is not None:
            self.my_perimeter_to_boolean_index[self.perimeter_border] = perimeter_border_boolean_index

        return result

    @computed_field  # type: ignore[misc]
    @property
    def perimeter_to_boolean_index(self) -> dict[BasePerimeter, NpNDArrayBool]:
        assert self.result is not None
        return self.my_perimeter_to_boolean_index

    @validate_call(config={"arbitrary_types_allowed": True})
    def plot(self, ax: Axes, video: Optional[VideoMetadata] = None, coordinates_as_pixels: bool = False) -> None:
        assert self.result is not None

        inside_perimeter_border_plot_scaled = self.inside_perimeter_border

        if video:
            inside_perimeter_border_plot_scaled = video.prepare_coordinates_for_plotting(
                inside_perimeter_border_plot_scaled, coordinates_as_pixels
            )
            if coordinates_as_pixels:
                video.upscaled_video.ax_ticks_metric_to_pixel(ax)

        self.perimeter.plot(ax=ax, coordinates_as_pixels=coordinates_as_pixels)
        self.perimeter_border.plot(ax=ax, coordinates_as_pixels=coordinates_as_pixels)

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
