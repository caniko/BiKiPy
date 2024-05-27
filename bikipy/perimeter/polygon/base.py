from abc import ABC
from functools import cached_property
from logging import getLogger
from typing import Any, ClassVar, Literal, Optional

import numpy as np
from matplotlib.axes import Axes
from pydantic import computed_field, field_validator
from pydantic_numpy.typing import Np1DArrayBool, Np2DArrayFp64, NpNDArray

from bikipy.math.confinement.polygon import parallel_point_inside_polygon
from bikipy.math.geometry import clockwise_sort_points
from bikipy.math.graph import Graph
from bikipy.math.vector import (
    nearest_point_on_line_segment_to_coordinates,
    ray_and_line_segment_intersection,
    rotate_vectors_with_angle,
)
from bikipy.perimeter.base import BaseSinglePerimeter
from bikipy.perimeter.circle.model import CircleFixedRadiusPerimeter
from bikipy.utils.collection_utils import project_mask_to_original

logger = getLogger(__name__)


class BasePolygonPerimeter(BaseSinglePerimeter, ABC):
    vertices_in_pixels: Np2DArrayFp64
    derived_meters_per_pixel_source: Optional[Literal["side"]] = None

    polygon_order: ClassVar[Optional[int]]

    @classmethod
    @property
    def fields_to_exclude_from_single_schema(cls) -> set[str]:
        result = super().fields_to_exclude_from_single_schema
        result.update(("vertices_in_pixels", "derived_meters_per_pixel_source"))
        return result

    @computed_field  # type: ignore[misc]
    @property
    def _to_hash(self) -> list:
        result = super()._to_hash
        result.append(self.vertices_in_pixels.data.tobytes())
        return result

    @field_validator("vertices_in_pixels")
    def vertices_polygon_order_validator(cls, value: Np2DArrayFp64):
        if cls.polygon_order and (n := len(value)) != int(cls.polygon_order):
            msg = (
                f"The polygon class is in the {cls.polygon_order}th order. However, "
                f"the current polygon is of the {n}th order"
            )
            raise ValueError(msg)
        return np.ascontiguousarray(clockwise_sort_points(value), dtype=float)

    def __getitem__(self, item: int):
        return self.vertices_in_meters[item]

    def __repr__(self):
        return super().__repr__() + f"\n\tvertices_in_pixels={self.vertices_in_meters}"

    @computed_field  # type: ignore[misc]
    @property
    def derived_meters_per_pixel(self) -> float:
        if self.derived_meters_per_pixel_source == "side":
            first_side = self.vertices_in_pixels.edge_lengths[0]
            if not np.allclose(first_side, self.vertices_in_pixels.edge_lengths[1:]):
                msg = "Sides of polygon are not equal, side cannot be used as source for computing meters_per_pixel"
                raise AttributeError(msg)

            return self.derived_meters_per_pixel_source_metric_length / first_side

    @computed_field  # type: ignore[misc]
    @property
    def centroid_meters(self) -> Np2DArrayFp64:
        return self.metric_graph.centroid

    @computed_field  # type: ignore[misc]
    @cached_property
    def vertices_in_meters(self) -> Np2DArrayFp64:
        return self.vertices_in_pixels * self.meters_per_pixel

    @computed_field  # type: ignore[misc]
    @cached_property
    def metric_graph(self) -> Graph:
        return Graph(vertices=self.vertices_in_meters)

    @computed_field  # type: ignore[misc]
    @cached_property
    def pixel_graph(self) -> Graph:
        return Graph(vertices=self.vertices_in_pixels)

    @computed_field  # type: ignore[misc]
    @cached_property
    def equilateral(self) -> bool:
        return np.all(
            np.apply_along_axis(
                np.isclose, 0, self.metric_graph.edge_lengths[0], self.metric_graph.edge_lengths[1:], atol=1.0e-4
            ),
            axis=1,
        )

    @computed_field  # type: ignore[misc]
    @cached_property
    def circle(self) -> CircleFixedRadiusPerimeter:
        return CircleFixedRadiusPerimeter(
            center_pixels=self.pixel_graph.centroid,
            radius_length_pixels=np.mean(self.pixel_graph.vertex_midpoint_distances_to_centroid),
            manual_video=self.video,
        )

    def closest_point_on_edge_to_coordinates(self, coordinates: Np2DArrayFp64, inspect: bool = False) -> Np2DArrayFp64:
        # Closest point on the index-respective edge along axis 0, and coordinates along 1.
        closest_edge_point_to_coordinates_matrix = np.array(
            [
                nearest_point_on_line_segment_to_coordinates(*line_segment_pair, coordinates)
                for line_segment_pair in self.metric_graph.vertex_pairs
            ]
        )
        # Distance of the coordinate from the previous matrix
        vector_matrix = coordinates - closest_edge_point_to_coordinates_matrix
        distance_matrix = np.linalg.norm(vector_matrix, axis=2)

        argsorted_distance = np.argsort(distance_matrix, axis=0)
        closest_boolean_index = argsorted_distance == 0

        result = closest_edge_point_to_coordinates_matrix[closest_boolean_index]

        if self.inspect_closest_point_on_edge:
            self.plot_closest_point_on_edge_to_coordinates(coordinates, result)

        return result

    def _compute_confinement_boolean_index(self, coordinates: Np2DArrayFp64) -> Np1DArrayBool:
        result = parallel_point_inside_polygon(coordinates, self.metric_graph.linked_vertices, merge_ends=False)
        return result

    def ray_intersects_on_polygon(
        self,
        ray_origins: Np2DArrayFp64,
        ray_directions: Np2DArrayFp64,
        return_points: bool = False,
    ) -> NpNDArray:
        result = np.array(
            [
                ray_and_line_segment_intersection(ray_origins, ray_directions, *line_segment_pair, return_points)
                for line_segment_pair in self.metric_graph.vertex_pairs
            ]
        )
        if not return_points:
            return np.any(result, axis=0)
        return result

    def filter_by_ray_direction_offset_filter(
        self,
        op_label: str,
        ray_start_points: Np2DArrayFp64,
        ray_travel_direction_points: Np2DArrayFp64,
        max_radians: float,
        extra_ax: Optional[Axes] = None,
        *,
        angular_resolution: int = 200,
    ) -> tuple[Np1DArrayBool, dict[str, Any]]:
        """
        Determine if the object is within the ray cone

        Emit rays from the point representing the region of interest, and checking for collisions
        with the perimeter.

        This problem is non-trivial for polygons. This is not the best solution in terms of speed for our application;
        nevertheless, it is quite robust and had the lowest implementation time.

        :param op_label:
        :param ray_start_points:
        :param ray_travel_direction_points:
        :param max_radians:
        :param extra_ax:
        :param angular_resolution:
        :return:
        """
        ray_vectors = ray_travel_direction_points - ray_start_points

        in_direct_los = self.ray_intersects_on_polygon(
            ray_travel_direction_points,
            ray_vectors,
        )
        if np.all(in_direct_los):
            return in_direct_los

        radians_to_check = np.linspace(-max_radians, max_radians, angular_resolution)

        not_in_direct_los = ~in_direct_los
        rotated_ray_vectors = rotate_vectors_with_angle(ray_vectors[not_in_direct_los], radians_to_check)

        in_tolerable_los = np.empty(rotated_ray_vectors.shape[:2])
        for i in range(angular_resolution * 2):
            in_tolerable_los[i] = self.ray_intersects_on_polygon(
                ray_travel_direction_points[not_in_direct_los],
                rotated_ray_vectors[i],
            )
        in_tolerable_los = np.any(in_tolerable_los, axis=0)
        result = project_mask_to_original(in_tolerable_los, in_direct_los) | in_direct_los

        return result, dict()

    def change_reference(self, new_reference: Np2DArrayFp64, makesense_image_name: Optional[str] = None):
        if self.reference_point is None:
            msg = "Reference without defining a reference for the source perimeter object is disallowed"
            raise AttributeError(msg)

        if np.all(self.reference_point == new_reference):
            logger.debug("The provided reference_point is identical to the current")
            return self

        if self.reference_point is not None and np.any(self.reference_point):
            return self.copy(
                update={
                    "vertices_in_pixels": self.vertices_in_pixels + new_reference - self.reference_point,
                    "manual_video": self.video,
                    "makesense_image_name": makesense_image_name,
                }
            )

        return self

    def plot_perimeter_on_ax(
        self,
        ax: Axes,
        coordinates_as_pixels: bool = False,
        with_resize: bool = True,
        x_pixel_offset: float = 0.0,
        y_pixel_offset: float = 0.0,
        **plot_kwargs,
    ) -> None:
        if x_pixel_offset or y_pixel_offset:
            vertices = self.vertices_in_pixels + np.array([x_pixel_offset, y_pixel_offset])
            if not coordinates_as_pixels:
                vertices *= self.meters_per_pixel
        elif coordinates_as_pixels:
            vertices = self.vertices_in_pixels
        elif not coordinates_as_pixels:
            vertices = self.vertices_in_meters
        else:
            raise RuntimeError

        if coordinates_as_pixels and with_resize:
            vertices *= self.video.image_resize_multiplier

        for index in range(len(vertices)):
            following_index = 0 if index + 1 == len(vertices) else index + 1

            corner_a = vertices[index]
            corner_b = vertices[following_index]
            ax.plot(
                *np.vstack((corner_a, corner_b)).T,
                # label=f"{self.label}{index}",     # Activate this when inspecting the sorting of edges
                **plot_kwargs,
            )

    def _add_label_to_str(self, in_string):
        if self.label:
            return f"{self.label} {in_string}"
        if self.int_id:
            return f"{self.int_id} {in_string}"
        return in_string


def init_polygon(vertices_in_meters: Np2DArrayFp64, **kwargs) -> BasePolygonPerimeter:
    match vertices_in_meters.shape[0]:  # polygon_order
        case 3:
            from bikipy.perimeter.polygon.triangle import TrianglePerimeter

            return TrianglePerimeter(vertices_in_pixels=vertices_in_meters, **kwargs)
        case 4:
            from bikipy.perimeter.polygon.rectangle import RectanglePerimeter

            return RectanglePerimeter(vertices_in_pixels=vertices_in_meters, **kwargs)
        case _:
            return BasePolygonPerimeter(vertices_in_pixels=vertices_in_meters, **kwargs)
