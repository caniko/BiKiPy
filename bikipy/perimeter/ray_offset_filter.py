from functools import cached_property

import numpy as np
from matplotlib.axes import Axes
from pydantic import computed_field
from pydantic_numpy.typing import Np1DArrayBool, Np1DArrayFp64, Np2DArrayFp64

from bikipy._constant import QUIVER_KWARGS
from bikipy.math.vector import radians_from_a_to_b, unit_vector, rotate_vectors_with_angle
from bikipy.plot.generic import boolean_index_colormap
from abc import ABC
from functools import cached_property

from pydantic import computed_field
from pydantic_numpy.typing import Np1DArrayBool, Np2DArrayFp64

from bikipy.core.compute import AbstractComputePerimeterBooleanIndex
from bikipy.math.vector import unit_vector
from bikipy.perimeter.base import BasePerimeter
from bikipy.utils.collection_utils import project_mask_to_original


class AbstractComputeRayOffsetFilter(AbstractComputePerimeterBooleanIndex, ABC):
    ray_start_points: Np2DArrayFp64
    ray_travel_direction_points: Np2DArrayFp64
    max_radians: float

    heuristic_data_sources = ("ray_start_points", "ray_travel_direction_points", "max_radians")
    heuristic_data_sources_all_required = True

    @computed_field  # type: ignore[misc]
    @property
    def perimeter_to_boolean_index(self) -> dict[BasePerimeter, Np1DArrayBool]:
        return {self.perimeter: self.result}

    @computed_field  # type: ignore[misc]
    @cached_property
    def offset_rays(self) -> Np2DArrayFp64:
        return unit_vector(self.ray_travel_direction_points - self.ray_start_points)


class ComputeRayOffsetFilterPolygon(AbstractComputeRayOffsetFilter):
    """
    Determine if the object is within the ray cone

    Emit rays from the point representing the region of interest, and checking for collisions
    with the perimeter.

    This problem is non-trivial for polygons. This is not the best solution in terms of speed for our application;
    nevertheless, it is quite robust and had the lowest implementation time.
    """
    angular_resolution: int = 100

    @computed_field  # type: ignore[misc]
    @cached_property
    def in_direct_los(self) -> Np1DArrayBool:
        return self.perimeter.ray_intersects_on_polygon(
            self.ray_travel_direction_points,
            self.offset_rays,
        )

    @computed_field  # type: ignore[misc]
    @cached_property
    def result(self) -> Np1DArrayBool:
        if np.all(self.in_direct_los):
            return self.in_direct_los

        radians_to_check = np.linspace(-self.max_radians, self.max_radians, self.angular_resolution)

        not_in_direct_los = ~self.in_direct_los
        rotated_ray_vectors = rotate_vectors_with_angle(self.offset_rays[not_in_direct_los], radians_to_check)

        in_tolerable_los = np.empty(rotated_ray_vectors.shape[:2])
        for i in range(self.angular_resolution * 2):
            in_tolerable_los[i] = self.perimeter.ray_intersects_on_polygon(
                self.ray_travel_direction_points[not_in_direct_los],
                rotated_ray_vectors[i],
            )
        in_tolerable_los = np.any(in_tolerable_los, axis=0)
        result = project_mask_to_original(in_tolerable_los, self.in_direct_los) | self.in_direct_los

        return result

    def plot(self, ax: Axes, *args, **kwargs) -> None:
        raise NotImplementedError()


class ComputeRayOffsetFilterCircleTriangle(AbstractComputeRayOffsetFilter):
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
