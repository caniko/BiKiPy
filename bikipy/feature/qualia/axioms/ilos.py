from functools import cached_property
from typing import Optional

from matplotlib.axes import Axes
from pydantic import computed_field
from pydantic_numpy.typing import Np1DArrayBool, Np2DArrayFp64

from bikipy._constant import QUIVER_KWARGS
from bikipy.core.compute import AbstractComputePerimeterBooleanIndex
from bikipy.core.video import VideoMetadata
from bikipy.feature.tolerance.single import single_node_tolerance_model
from bikipy.math.vector import unit_vector
from bikipy.perimeter.base import BasePerimeter


class ComputeInLineOfSight(AbstractComputePerimeterBooleanIndex):
    ray_start_points: Np2DArrayFp64
    ray_travel_direction_points: Np2DArrayFp64
    max_radians: float

    heuristic_data_sources = ("ray_start_points", "ray_travel_direction_points", "max_radians")
    heuristic_data_sources_all_required = True

    @computed_field  # type: ignore[misc]
    @cached_property
    def result(self) -> Np1DArrayBool:
        result = self.perimeter.ray_direction_filter(
            "InLineOfSight", self.ray_start_points, self.ray_travel_direction_points, self.max_radians
        )
        if self.tolerance_modelling:
            result = single_node_tolerance_model(result, self.fps)
        return result

    @computed_field  # type: ignore[misc]
    @property
    def perimeter_to_boolean_index(self) -> dict[BasePerimeter, Np1DArrayBool]:
        return {self.perimeter: self.result}

    def plot(self, ax: Axes, video: Optional[VideoMetadata] = None, coordinates_as_pixels: bool = False) -> None:
        if video:
            if coordinates_as_pixels:
                video.upscaled_video.ax_ticks_metric_to_pixel(ax)

            ray_travel_direction_points = video.prepare_coordinates_for_plotting(
                self.ray_travel_direction_points, coordinates_as_pixels
            )
            ray_start_points = video.prepare_coordinates_for_plotting(self.ray_start_points, coordinates_as_pixels)
        else:
            ray_travel_direction_points = self.ray_travel_direction_points
            ray_start_points = self.ray_start_points

        ray_vectors = unit_vector(ray_travel_direction_points - ray_start_points)

        ax.quiver(
            *ray_travel_direction_points[self.result].T,
            *ray_vectors[self.result].T,
            label="Valid",
            color="b",
            **QUIVER_KWARGS,
        )

        not_result = ~self.result
        ax.quiver(
            *ray_travel_direction_points[not_result].T,
            *ray_vectors[not_result].T,
            label="Invalid",
            color="r",
            **QUIVER_KWARGS,
        )

        self.perimeter.plot_perimeter_on_ax(ax=ax, coordinates_as_pixels=coordinates_as_pixels)

        self.plot_finalization(ax)
