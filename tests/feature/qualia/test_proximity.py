from functools import cached_property
from logging import getLogger
from typing import Optional

from matplotlib.axes import Axes
from pydantic import computed_field, validate_call
from pydantic_numpy.typing import NpNDArrayBool, NpNDArrayFp64

from bikipy import runtime_settings
from bikipy.core.compute import AbstractComputePerimeterBooleanIndex
from bikipy.core.video import VideoMetadata
from bikipy.feature.tolerance.single import single_node_tolerance_model
from bikipy.perimeter.base import SinglePerimeter
from bikipy.utils.plot import BOTTOM_LEGEND_KWARGS
from bikipy.utils.plot.color import make_color_map

logger = getLogger(__name__)


class ComputeProximity(AbstractComputePerimeterBooleanIndex):
    perimeter: SinglePerimeter
    maximum_distance: float | NpNDArrayFp64
    inside_perimeter_border: NpNDArrayFp64
    outside_perimeter_border: Optional[NpNDArrayFp64] = None
    outside_perimeter: Optional[NpNDArrayBool] = None

    @computed_field  # type: ignore[misc]
    @cached_property
    def result(self) -> NpNDArrayBool:
        self.perimeter_border = self.perimeter.expand(self.maximum_distance)
        self.inside_perimeter_border = self.perimeter_border.compute_confinement_boolean_index(
            coordinates=self.inside_perimeter_border
        )
        result = self.inside_perimeter_border

        if self.perimeter.impenetrable:
            self.outside_impenetrable_bi = ~self.perimeter.compute_confinement_boolean_index(
                coordinates=self.inside_perimeter_border
            )
            result = result & self.outside_impenetrable_bi

        if self.outside_perimeter_border is not None:
            self.outside_perimeter_border_bi = ~self.perimeter_border.compute_confinement_boolean_index(
                self.outside_perimeter_border
            )
            result = result & self.outside_perimeter_border_bi

        if self.outside_perimeter is not None:
            self.outside_perimeter_bi = ~self.perimeter.compute_confinement_boolean_index(self.outside_perimeter_border)
            result = result & self.outside_perimeter_bi

        if self.tolerance_modelling:
            result = single_node_tolerance_model(result, self.video.fps)

        return result

    @validate_call(config={"arbitrary_types_allowed": True})
    def plot(self, ax: Axes, video: VideoMetadata, coordinates_as_pixels: bool = False) -> None:
        assert self.result is not None

        inside_perimeter_border_plot_scaled = video.prepare_coordinates_for_plotting(
            self.inside_perimeter_border, coordinates_as_pixels
        )

        if coordinates_as_pixels:
            video.upscaled_video.ax_ticks_metric_to_pixel(ax)

        ax.set_title(self.label)

        self.perimeter.plot(ax=ax, coordinates_as_pixels=coordinates_as_pixels)
        self.perimeter_border.plot(ax=ax, coordinates_as_pixels=coordinates_as_pixels)

        color_count = 1
        if self.perimeter.impenetrable:
            color_count += 1
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
        if self.perimeter.impenetrable:
            ax.scatter(
                *inside_perimeter_border_plot_scaled[self.outside_impenetrable_bi & not_result].T,
                marker="x",
                alpha=runtime_settings.matplotlib_scatter_alpha,
                label="Outside impenetrable valid; other invalid",
                color=next(color_map_iter),
            )
        if self.outside_perimeter_border is not None:
            ax.scatter(
                *inside_perimeter_border_plot_scaled[self.outside_perimeter_border_bi & not_result].T,
                marker="x",
                alpha=runtime_settings.matplotlib_scatter_alpha,
                label="Outside border valid; other invalid",
                color=next(color_map_iter),
            )
        if self.outside_perimeter is not None:
            ax.scatter(
                *inside_perimeter_border_plot_scaled[self.outside_perimeter_bi & not_result].T,
                marker="x",
                alpha=runtime_settings.matplotlib_scatter_alpha,
                label="Outside perimeter valid; other invalid",
                color=next(color_map_iter),
            )

        ax.legend(**BOTTOM_LEGEND_KWARGS)
