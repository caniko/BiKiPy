from functools import cached_property
from typing import Optional

from matplotlib.axes import Axes
from pydantic_numpy.dtype import NDArrayFp64

from bikipy import runtime_settings
from bikipy.core.video import VideoMetadata
from bikipy.feature.compute import AbstractComputeBooleanIndex
from bikipy.feature.tolerance.single import single_node_tolerance_model
from bikipy.perimeter.base import SinglePerimeter
from bikipy.utils.math.vector import unit_vector


class ComputeInLineOfSight(AbstractComputeBooleanIndex):
    perimeter: SinglePerimeter = ...
    ray_start_point: NDArrayFp64 = ...
    ray_travel_direction_point: NDArrayFp64 = ...
    max_radians: float = ...

    @cached_property
    def result(self):
        result = self.perimeter.ray_direction_filter(
            self.ray_start_point, self.ray_travel_direction_point, self.max_radians
        )
        if self.tolerance_modelling:
            result = single_node_tolerance_model(result, self.video.fps)
        return result

    def plot(self, ax: Axes, video: Optional[VideoMetadata] = None, inspect_pixels: bool = False) -> None:
        ray_travel_direction_point = (
            video.prepare_coordinates_for_plotting(self.ray_travel_direction_point, inspect_pixels)
            if video
            else self.ray_travel_direction_point
        )
        ray_vectors = ray_travel_direction_point - self.ray_start_point

        if video:
            ray_vectors = video.prepare_coordinates_for_plotting(unit_vector(ray_vectors), inspect_pixels) * 0.025
            if inspect_pixels:
                video.upscaled_video.ax_ticks_metric_to_pixel(ax)

        self.perimeter.plot(inspect_pixels=inspect_pixels, ax=ax)

        quiver_kwargs = {
            "angles": "xy",
            "scale_units": "dots",
            "scale": 1.0,
            "alpha": runtime_settings.matplotlib_scatter_alpha,
        }

        ax.quiver(
            *ray_travel_direction_point[self.result].T,
            *ray_vectors[self.result].T,
            label="Valid",
            color="b",
            **quiver_kwargs,
        )

        not_result = ~self.result
        ax.quiver(
            *ray_travel_direction_point[not_result].T,
            *ray_vectors[not_result].T,
            label="Invalid",
            color="r",
            **quiver_kwargs,
        )

        self.plot_finalization(ax, video)
