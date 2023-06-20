from abc import ABC
from functools import cached_property
from logging import getLogger
from typing import ClassVar, Literal, Optional, TypeVar

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as nt
from matplotlib.axes import Axes
from pydantic import validator
from pydantic_numpy.dtype import NDArrayFp64

from bikipy.core.video import VideoMetadata
from bikipy.perimeter.base import BaseSinglePerimeter
from bikipy.perimeter.circle.model import CircleFixedRadiusPerimeter
from bikipy.utils.collection_utils import (
    evenly_spaced_indices_from_sequence,
    project_mask_to_original,
)
from bikipy.utils.graph import Graph
from bikipy.utils.math.confinement.polygon import parallel_point_inside_polygon
from bikipy.utils.math.geometry import clockwise_sort_points
from bikipy.utils.math.vector import (
    nearest_point_on_line_segment_to_coordinates,
    ray_and_line_segment_intersection,
    rotate_vectors_with_angle,
    unit_vector,
)
from bikipy.utils.plot import BOTTOM_LEGEND_KWARGS, TIGHT_LAYOUT_KWARGS

logger = getLogger(__name__)


class BasePolygonPerimeter(BaseSinglePerimeter, ABC):
    vertices_in_pixels: NDArrayFp64 = ...
    derived_meters_per_pixel_source: Optional[Literal["side"]]

    polygon_order: ClassVar[Optional[int]]

    @classmethod
    @property
    def schemantic_fields_to_exclude_from_config_schema(cls) -> set[str]:
        result = super().schemantic_fields_to_exclude_from_config_schema
        result.update(("vertices_in_pixels", "derived_meters_per_pixel_source"))
        return result

    @property
    def _to_hash(self) -> list:
        result = super()._to_hash
        result.append(self.vertices_in_pixels.data.tobytes())
        return result

    @validator("vertices_in_pixels")
    def vertices_polygon_order_validator(cls, value: NDArrayFp64):
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

    @property
    def derived_meters_per_pixel(self) -> float:
        if self.derived_meters_per_pixel_source == "side":
            first_side = self.vertices_in_pixels.edge_lengths[0]
            if not np.allclose(first_side, self.vertices_in_pixels.edge_lengths[1:]):
                msg = "Sides of polygon are not equal, side cannot be used as source for computing meters_per_pixel"
                raise AttributeError(msg)

            return self.derived_meters_per_pixel_source_metric_length / first_side

    @property
    def centroid_meters(self) -> np.ndarray[float, np.dtype[np.float64]]:
        return self.metric_graph.centroid

    @cached_property
    def vertices_in_meters(self) -> np.ndarray[float, np.dtype[np.float64]]:
        return self.vertices_in_pixels * self.video.meters_per_pixel

    @cached_property
    def metric_graph(self) -> Graph:
        return Graph(vertices=self.vertices_in_meters)

    @cached_property
    def pixel_graph(self) -> Graph:
        return Graph(vertices=self.vertices_in_pixels)

    @cached_property
    def equilateral(self) -> bool:
        return np.all(
            np.apply_along_axis(
                np.isclose, 0, self.metric_graph.edge_lengths[0], self.metric_graph.edge_lengths[1:], atol=1.0e-4
            ),
            axis=1,
        )

    @cached_property
    def circle(self):
        return CircleFixedRadiusPerimeter(
            center_pixels=self.pixel_graph.centroid,
            radius_pixels=np.mean(self.pixel_graph.vertex_midpoint_distances_to_centroid),
            manual_video=self.video,
        )

    def closest_point_on_edge_to_coordinates(
        self, coordinates: NDArrayFp64, inspect: bool = False
    ) -> np.ndarray[float, np.dtype[np.float64]]:
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

        if inspect:
            indexable_t = closest_edge_point_to_coordinates_matrix.transpose(1, 2, 0)

            for i in evenly_spaced_indices_from_sequence(coordinates, 9):
                fig, ax = self.video.subplot()
                to_skip = []
                for y, point in enumerate(indexable_t[i].T):
                    if y in to_skip:
                        continue
                    duplicates_boolean_indices = np.all(
                        np.apply_along_axis(np.isclose, 0, point, indexable_t[i].T, atol=1.0e-4),
                        axis=1,
                    )
                    sort_indices = ", ".join(argsorted_distance.T[i][duplicates_boolean_indices].astype(str))
                    ax.scatter(*point, label=sort_indices)

                    to_skip.extend(np.where(duplicates_boolean_indices)[0].tolist())

                ax.scatter(*coordinates[i], label="coordinate")
                fig.legend(**BOTTOM_LEGEND_KWARGS)
                fig.tight_layout(**TIGHT_LAYOUT_KWARGS)
                plt.show()

        return result

    def vector_to_closest_point_on_edge(
        self,
        coordinates: NDArrayFp64,
        closest_point_on_edge_to_coordinates: Optional[NDArrayFp64] = None,
    ) -> np.ndarray[float, np.dtype[np.float64]]:
        if closest_point_on_edge_to_coordinates is None:
            closest_point_on_edge_to_coordinates = self.closest_point_on_edge_to_coordinates(coordinates)
        return unit_vector(closest_point_on_edge_to_coordinates - coordinates)

    def compute_confined_coordinate_boolean_index(
        self, coordinates: NDArrayFp64, manual_video: Optional[VideoMetadata] = None, ax: Axes = None, **inspect_kwargs
    ) -> np.ndarray[bool, bool]:
        result = parallel_point_inside_polygon(coordinates, self.metric_graph.linked_vertices, merge_ends=False)

        self._post_confinement_analysis_inspect_plot(result, coordinates, manual_video, ax, **inspect_kwargs)

        return result

    def ray_intersects_on_polygon(
        self,
        ray_origins: NDArrayFp64,
        ray_directions: NDArrayFp64,
        return_points: bool = False,
    ) -> nt.NDArray:
        result = np.array(
            [
                ray_and_line_segment_intersection(ray_origins, ray_directions, *line_segment_pair, return_points)
                for line_segment_pair in self.metric_graph.vertex_pairs
            ]
        )
        if not return_points:
            return np.any(result, axis=0)
        return result

    def ray_direction_filter(
        self,
        ray_start_point: NDArrayFp64,
        ray_travel_direction_point: NDArrayFp64,
        max_radians: float,
        angular_resolution: int = 400,
    ) -> np.ndarray[bool, bool]:
        """
        Determine if the object is within the ray cone

        This problem is called the "in line of sight" (ilos) problem, and is non-trivial. This is not the best solution
        in terms of speed for our application; nevertheless, it is quite robust and had the lowest implementation time.
        The solution is to emit rays from the point representing the region of interest, and checking for collisions
        with the perimeter.

        :param ray_travel_direction_point:
        :param ray_start_point:
        :param max_radians:
        :param angular_resolution:
        :return:
        """
        ray_vectors = ray_travel_direction_point - ray_start_point

        in_direct_los = self.ray_intersects_on_polygon(
            ray_travel_direction_point,
            ray_vectors,
        )
        if np.all(in_direct_los):
            return in_direct_los

        positive_angles = np.linspace(max_radians, 0.0, angular_resolution)
        negative_angles = -positive_angles
        angles = np.concatenate([negative_angles, positive_angles])

        not_in_direct_los = ~in_direct_los
        rotated_ray_vectors = rotate_vectors_with_angle(ray_vectors[not_in_direct_los], angles)

        in_tolerable_los = np.empty(rotated_ray_vectors.shape[:2])
        for i in range(angular_resolution * 2):
            in_tolerable_los[i] = self.ray_intersects_on_polygon(
                ray_travel_direction_point[not_in_direct_los],
                rotated_ray_vectors[i],
            )
        in_tolerable_los = np.any(in_tolerable_los, axis=0)
        result = project_mask_to_original(in_tolerable_los, in_direct_los) | in_direct_los

        return result

    def change_reference(self, new_reference: NDArrayFp64, makesense_image_name: Optional[str] = None):
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
        self, ax: Axes, inspect_pixels: bool = False, manual_resize_multiplier: Optional[float] = None, **plot_kwargs
    ) -> None:
        vertices = self.vertices_in_pixels if inspect_pixels else self.vertices_in_meters

        if inspect_pixels:
            vertices = vertices * (manual_resize_multiplier or self.video.image_resize_multiplier)

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


PolygonPerimeter = TypeVar("PolygonPerimeter", bound=BasePolygonPerimeter)


def init_polygon(vertices_in_meters: NDArrayFp64, **kwargs) -> PolygonPerimeter:
    match vertices_in_meters.shape[0]:  # polygon_order
        case 3:
            from bikipy.perimeter.polygon.triangular import TriangularPerimeter

            return TriangularPerimeter(vertices_in_pixels=vertices_in_meters, **kwargs)
        case 4:
            from bikipy.perimeter.polygon.rectangle import RectanglePerimeter

            return RectanglePerimeter(vertices_in_pixels=vertices_in_meters, **kwargs)
        case _:
            return BasePolygonPerimeter(vertices_in_pixels=vertices_in_meters, **kwargs)
