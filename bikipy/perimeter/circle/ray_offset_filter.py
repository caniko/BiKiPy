from abc import ABC
from functools import cached_property
from typing import Any, Optional, Self

import numpy as np
from matplotlib.axes import Axes
from pydantic import computed_field
from pydantic_numpy.typing import Np1DArrayBool, Np2DArrayFp64, Np1DArrayFp64

from bikipy._constant import QUIVER_KWARGS
from bikipy.core.compute import AbstractComputePerimeterBooleanIndex
from bikipy.core.video import VideoMetadata
from bikipy.feature.qualia.axioms.ray_offset_filter import ComputeRayOffsetFilter
from bikipy.feature.tolerance.single import single_node_tolerance_model
from bikipy.math.vector import unit_vector, radians_from_a_to_b
from bikipy.perimeter.base import BasePerimeter
from bikipy.perimeter.circle.model import BaseCirclePerimeter
from bikipy.perimeter.polygon.base import BasePolygonPerimeter
from bikipy.perimeter.polygon.triangle import TrianglePerimeter
from bikipy.plot.generic import boolean_index_colormap


class ComputeRayOffsetFilterCircleTriangle(ComputeRayOffsetFilter):
    @computed_field  # type: ignore[misc]
    @cached_property
    def closest_points_on_edges(self) -> Np2DArrayFp64:
        return self.perimeter.closest_point_on_edge_to_coordinates(self.ray_travel_direction_points)

    @computed_field  # type: ignore[misc]
    @cached_property
    def vector_to_closest_point_on_edge(self) -> Np2DArrayFp64:
        return self.perimeter.vector_to_closest_point_on_edge(
            self.ray_travel_direction_points, self.closest_points_on_edges
        )

    @computed_field  # type: ignore[misc]
    @cached_property
    def offset_rays(self) -> Np2DArrayFp64:
        return unit_vector(self.ray_travel_direction_points - self.ray_start_points)

    @computed_field  # type: ignore[misc]
    @cached_property
    def normal_to_ray_radians_offset(self) -> Np1DArrayFp64:
        return radians_from_a_to_b(self.vector_to_closest_point_on_edge, self.offset_rays)

    @computed_field  # type: ignore[misc]
    @cached_property
    def result(self) -> Np1DArrayBool:
        return self.normal_to_ray_radians_offset <= self.max_radians

    @computed_field  # type: ignore[misc]
    @cached_property
    def plot_ray_travel_direction_points(self) -> np.ndarray:
        return self.video.prepare_coordinates_for_plotting(self.ray_travel_direction_points, step=True)

    @computed_field  # type: ignore[misc]
    @cached_property
    def plot_travel_direction_rays(self) -> np.ndarray:
        return self.video.prepare_coordinates_for_plotting(self.travel_direction_rays, step=True)

    @computed_field  # type: ignore[misc]
    @cached_property
    def plot_vectors_to_closest_point_on_edge(self) -> np.ndarray:
        return self.video.prepare_coordinates_for_plotting(self.vector_to_closest_point_on_edge, step=True)

    @computed_field  # type: ignore[misc]
    @cached_property
    def plot_normal_to_ray_degree_offset(self) -> np.ndarray:
        return np.rad2deg(self.normal_to_ray_radians_offset[self.video.plot_slice])

    def plot(self, ax: Axes, **kwargs) -> None:
        for (
            color,
            ray_travel_direction_point,
            travel_direction_ray,
            vector_to_closest_point_on_edge,
            degrees,
        ) in zip(
            boolean_index_colormap(self.result[self.video.plot_slice]),
            self.plot_ray_travel_direction_points,
            self.plot_travel_direction_rays,
            self.plot_vectors_to_closest_point_on_edge,
            self.plot_normal_to_ray_degree_offset,
        ):
            rtdx, rtdy = ray_travel_direction_point

            ax.quiver(rtdx, rtdy, *travel_direction_ray, color=color, **QUIVER_KWARGS)
            ax.quiver(rtdx, rtdy, *vector_to_closest_point_on_edge, color="green", **QUIVER_KWARGS)
            ax.text(rtdx, rtdy, f"{degrees:.1f}°", fontsize=8, color="green")
