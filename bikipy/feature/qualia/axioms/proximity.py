from functools import cached_property
from typing import Final, Optional, Self

from matplotlib.axes import Axes
from pydantic import computed_field, model_validator, validate_call
from pydantic_numpy.typing import Np1DArrayBool, NpNDArrayFp64

from bikipy import runtime_settings
from bikipy.core.compute import AbstractComputePerimeterBooleanIndex
from bikipy.core.video import VideoMetadata
from bikipy.perimeter.base import BasePerimeter, BaseSinglePerimeter
from bikipy.utils.plot.color import make_color_map

_PROXIMITY_COMPUTE_OP_LABEL: Final[str] = "compute_proximity-{}"


class ComputeProximity(AbstractComputePerimeterBooleanIndex):
    maximum_distance: float | NpNDArrayFp64

    inside_perimeter: Optional[NpNDArrayFp64] = None
    outside_perimeter: Optional[NpNDArrayFp64] = None
    inside_perimeter_border: Optional[NpNDArrayFp64] = None
    outside_perimeter_border: Optional[NpNDArrayFp64] = None

    inside_perimeter_boolean_index: Optional[Np1DArrayBool] = None
    outside_perimeter_boolean_index: Optional[Np1DArrayBool] = None
    inside_perimeter_border_boolean_index: Optional[Np1DArrayBool] = None
    outside_perimeter_border_boolean_index: Optional[Np1DArrayBool] = None

    heuristic_data_sources = (
        "inside_perimeter",
        "outside_perimeter",
        "inside_perimeter_border",
        "outside_perimeter_border",
    )

    @model_validator(mode="after")
    def perform_computation(self) -> Self:
        if self.inside_perimeter is not None and self.inside_perimeter_boolean_index is None:
            self.inside_perimeter_boolean_index = self.perimeter.confinement_boolean_index(
                _PROXIMITY_COMPUTE_OP_LABEL.format("inside"), self.inside_perimeter
            )

        if self.outside_perimeter is not None and self.outside_perimeter_boolean_index is None:
            self.outside_perimeter_boolean_index = ~self.perimeter.confinement_boolean_index(
                _PROXIMITY_COMPUTE_OP_LABEL.format("outside"), self.outside_perimeter
            )

        if self.inside_perimeter_border is not None and self.inside_perimeter_border_boolean_index is None:
            self.inside_perimeter_border_boolean_index = self.perimeter_border.confinement_boolean_index(
                _PROXIMITY_COMPUTE_OP_LABEL.format("inside-border"), self.inside_perimeter_border
            )

        if self.outside_perimeter_border is not None and self.outside_perimeter_border_boolean_index is None:
            self.outside_perimeter_border_boolean_index = ~self.perimeter_border.confinement_boolean_index(
                _PROXIMITY_COMPUTE_OP_LABEL.format("outside-border"), self.outside_perimeter_border
            )
        return self

    @computed_field  # type: ignore[misc]
    @cached_property
    def perimeter_border(self) -> BaseSinglePerimeter:
        return self.perimeter.expand(self.maximum_distance)

    @computed_field  # type: ignore[misc]
    @cached_property
    def result(self) -> Np1DArrayBool:
        if self.valid_border is not None and self.valid_perimeter is not None:
            return self.valid_border & self.valid_perimeter

        if self.valid_border is not None:
            return self.valid_border

        if self.valid_perimeter is not None:
            return self.valid_perimeter

        assert False

    @computed_field  # type: ignore[misc]
    @cached_property
    def valid_border(self) -> Np1DArrayBool | None:
        if (
            self.outside_perimeter_border_boolean_index is not None
            and self.inside_perimeter_border_boolean_index is not None
        ):
            return self.outside_perimeter_border_boolean_index & self.inside_perimeter_border_boolean_index

        if self.outside_perimeter_border_boolean_index is not None:
            return self.outside_perimeter_border_boolean_index

        if self.inside_perimeter_border_boolean_index is not None:
            return self.inside_perimeter_border_boolean_index

    @computed_field  # type: ignore[misc]
    @cached_property
    def valid_perimeter(self) -> Np1DArrayBool | None:
        if self.outside_perimeter_boolean_index is not None and self.inside_perimeter_boolean_index is not None:
            return self.outside_perimeter_boolean_index & self.inside_perimeter_boolean_index

        if self.outside_perimeter_boolean_index is not None:
            return self.outside_perimeter_boolean_index

        if self.inside_perimeter_boolean_index is not None:
            return self.inside_perimeter_boolean_index

    @computed_field(repr=False)  # type: ignore[misc]
    @property
    def perimeter_to_boolean_index(self) -> dict[BasePerimeter, Np1DArrayBool]:
        result = {}
        if self.valid_border is not None:
            result[self.perimeter_border] = self.valid_border

        if self.valid_perimeter is not None:
            result[self.perimeter] = self.valid_perimeter
        return result

    @validate_call(config=dict(arbitrary_types_allowed=True))
    def plot(self, ax: Axes, video: Optional[VideoMetadata] = None, coordinates_as_pixels: bool = False) -> None:
        assert self.result is not None

        inside_perimeter_border_plot_scaled = self.inside_perimeter_border

        if video:
            inside_perimeter_border_plot_scaled = video.prepare_coordinates_for_plotting(
                inside_perimeter_border_plot_scaled, coordinates_as_pixels
            )
            if coordinates_as_pixels:
                video.upscaled_video.ax_ticks_metric_to_pixel(ax)

        self.perimeter.plot_perimeter_on_ax(ax=ax, coordinates_as_pixels=coordinates_as_pixels)
        self.perimeter_border.plot_perimeter_on_ax(ax=ax, coordinates_as_pixels=coordinates_as_pixels)

        color_count = 1
        if self.valid_border is not None:
            color_count += 1
        if self.valid_perimeter is not None:
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
        if self.valid_border is not None:
            ax.scatter(
                *inside_perimeter_border_plot_scaled[self.valid_border & not_result].T,
                marker="x",
                alpha=runtime_settings.matplotlib_scatter_alpha,
                label="ValidBorder",
                color=next(color_map_iter),
            )
        if self.valid_perimeter is not None:
            ax.scatter(
                *inside_perimeter_border_plot_scaled[self.valid_perimeter & not_result].T,
                marker="x",
                alpha=runtime_settings.matplotlib_scatter_alpha,
                label="ValidPerimeter",
                color=next(color_map_iter),
            )

        self.plot_finalization(ax)
