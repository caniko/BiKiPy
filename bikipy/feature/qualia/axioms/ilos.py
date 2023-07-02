from functools import cached_property
from typing import Optional

import numpy as np
from matplotlib.axes import Axes
from pydantic_numpy.dtype import NDArrayFp64

from bikipy import runtime_settings
from bikipy.core.compute import AbstractComputePerimeterBooleanIndex
from bikipy.core.video import VideoMetadata
from bikipy.feature.tolerance.single import single_node_tolerance_model
from bikipy.perimeter.base import Perimeter
from bikipy.utils.math.vector import unit_vector


class ComputeInLineOfSight(AbstractComputePerimeterBooleanIndex):
    ray_start_point: NDArrayFp64 = ...
    ray_travel_direction_point: NDArrayFp64 = ...
    max_radians: float = ...

    manual_ray_vectors: Optional[NDArrayFp64]

    heuristic_data_sources = ("ray_start_point", "ray_travel_direction_point", "max_radians")
    heuristic_data_sources_all_required = True

    @cached_property
    def result(self):
        result = self.perimeter.ray_direction_filter(
            self.ray_start_point, self.ray_travel_direction_point, self.max_radians
        )
        if self.tolerance_modelling:
            result = single_node_tolerance_model(result, self.video.fps)
        return result

    @property
    def perimeter_to_boolean_index(self) -> dict[Perimeter, np.ndarray[bool, bool]]:
        return {self.perimeter: self.result}

    def plot(self, ax: Axes, video: Optional[VideoMetadata] = None, coordinates_as_pixels: bool = False) -> None:
        ray_travel_direction_point = (
            video.prepare_coordinates_for_plotting(self.ray_travel_direction_point, coordinates_as_pixels)
            if video
            else self.ray_travel_direction_point
        )
        ray_vectors = self.manual_ray_vectors or (ray_travel_direction_point - self.ray_start_point)

        if video:
            ray_vectors = (
                video.prepare_coordinates_for_plotting(unit_vector(ray_vectors), coordinates_as_pixels) * 0.025
            )
            if coordinates_as_pixels:
                video.upscaled_video.ax_ticks_metric_to_pixel(ax)

        self.perimeter.plot(coordinates_as_pixels=coordinates_as_pixels, ax=ax)

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
