from abc import ABC
from functools import cached_property
from logging import getLogger
from typing import Any, ClassVar, Optional, TypeVar

import matplotlib.pyplot as plt
import numpy as np
from pydantic import FilePath, validator
from pydantic_numpy import NDArray

from bikipy.core.typing import NDArrayFp64, NDArrayInt16, NDArrayBool
from bikipy.core.video import convert_meters_to_pixels
from bikipy.perimeter.base import BaseSinglePerimeter
from bikipy.perimeter.radial.circle import CirclePerimeter
from bikipy.utils.collection_utils import (
    evenly_spaced_indices_from_sequence,
    project_mask_to_original,
)
from bikipy.utils.graph import Graph
from bikipy.utils.math.geometry import clockwise_sort_points
from bikipy.utils.math.point_in_polygon import parallel_point_in_polygon
from bikipy.utils.math.vector import (
    nearest_point_on_line_segment_to_coordinates,
    unit_vector,
    ray_and_line_segment_intersection,
    rotate_vectors_with_angle,
)
from bikipy.utils.plotting import generic_inspection_finalization

logger = getLogger(__name__)


class BasePolygonPerimeter(BaseSinglePerimeter, ABC):
    vertices_in_pixels: NDArrayFp64 = ...

    category = "perimeter"

    polygon_order: ClassVar[Optional[int]]

    @classmethod
    @property
    def exclude_from_settings_schema(cls) -> set[str]:
        """
        Some required fields for a class are sometimes highly specific to its respective object. These fields should
        be recorded in this class-property to be excluded by the settings generator function in the ingress module
        :return:
        """
        return super().exclude_from_settings_schema.union({"vertices_in_pixels"})

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
    def centroid_meters(self) -> NDArrayFp64:
        return self.metric_graph.centroid

    @cached_property
    def vertices_in_meters(self) -> NDArrayFp64:
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
        return CirclePerimeter(
            center_pixels=self.pixel_graph.centroid_meters,
            radius_meters=np.mean(self.metric_graph.vertex_midpoint_distances_to_centroid),
            manual_video=self.video,
        )

    def closest_point_on_edge_to_coordinates(self, coordinates: NDArrayFp64, inspect: bool = False) -> NDArrayFp64:
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
                fig, ax = plt.subplots()
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
                plt.legend()
                plt.tight_layout()
                plt.show()

        return result

    def vector_to_closest_point_on_edge(
        self,
        coordinates: NDArrayFp64,
        closest_point_on_edge_to_coordinates: Optional[NDArrayFp64] = None,
    ) -> NDArrayFp64:
        if closest_point_on_edge_to_coordinates is None:
            closest_point_on_edge_to_coordinates = self.closest_point_on_edge_to_coordinates(coordinates)
        return unit_vector(closest_point_on_edge_to_coordinates - coordinates)

    def confined_coordinate_boolean_index(self, coordinates: NDArrayFp64) -> NDArrayBool:
        return parallel_point_in_polygon(
            coordinates, self.metric_graph.linked_vertices, merge_ends=False, inspect_arg=self.class_inspect_arg
        )

    def ray_intersects_on_polygon(
        self,
        ray_origins: NDArrayFp64,
        ray_directions: NDArrayFp64,
        return_points: bool = False,
    ) -> NDArray:
        result = np.array(
            [
                ray_and_line_segment_intersection(ray_origins, ray_directions, *line_segment_pair, return_points)
                for line_segment_pair in self.metric_graph.vertex_pairs
            ]
        )
        if not return_points:
            return np.any(result, axis=0)
        return result

    def closest_ray_intersection_points(
        self,
        ray_origins: NDArrayFp64,
        ray_directions: NDArrayFp64,
    ) -> NDArrayFp64:
        """
        The first intersection point between a ray, and the polygon

        :return:
        """
        ray_intersection_points_on_polygon = self.ray_intersects_on_polygon(ray_origins, ray_directions)
        vector_matrix = ray_origins - ray_intersection_points_on_polygon
        distance_matrix = np.linalg.norm(vector_matrix, axis=2)

        argsorted_distance = np.argsort(distance_matrix, axis=0)
        closest_boolean_index = argsorted_distance == 0

        return ray_intersection_points_on_polygon[closest_boolean_index]

    def gaze_direction_filter(
        self,
        gaze_travel_direction_point: NDArrayFp64,
        gaze_start_point: NDArrayFp64,
        max_radians: float,
        angular_resolution: int = 400,
        manual_ax: Any = None,
        **inspect_kwargs,
    ) -> NDArrayBool:
        """
        Determine if the object is within the gaze cone

        This problem is called the "in line of sight" (ilos) problem, and is non-trivial. This is not the best solution
        in terms of speed for our application; nevertheless, it is quite robust and had the lowest implementation time.
        The solution is to emit rays from

        :param gaze_travel_direction_point:
        :param gaze_start_point:
        :param max_radians:
        :param angular_resolution:
        :param inspect_kwargs:
        :return:
        """
        gaze_vectors = gaze_travel_direction_point - gaze_start_point

        in_direct_los = self.ray_intersects_on_polygon(
            gaze_travel_direction_point,
            gaze_vectors,
        )
        if np.all(in_direct_los):
            return in_direct_los

        positive_angles = np.linspace(max_radians, 0.0, angular_resolution)
        negative_angles = -positive_angles
        angles = np.concatenate([negative_angles, positive_angles])

        not_in_direct_los = ~in_direct_los
        rotated_gaze_vectors = rotate_vectors_with_angle(gaze_vectors[not_in_direct_los], angles)

        in_tolerable_los = np.empty(rotated_gaze_vectors.shape[:2])
        for i in range(angular_resolution * 2):
            in_tolerable_los[i] = self.ray_intersects_on_polygon(
                gaze_travel_direction_point[not_in_direct_los],
                rotated_gaze_vectors[i],
            )
        in_tolerable_los = np.any(in_tolerable_los, axis=0)
        result = project_mask_to_original(in_tolerable_los, in_direct_los) | in_direct_los

        if self.inspect_arg or manual_ax:
            from bikipy.feature.attention.gaze import gaze_inspection_plot

            gaze_inspection_plot(
                self,
                result,
                gaze_vectors,
                gaze_travel_direction_point,
                manual_ax=manual_ax,
                **inspect_kwargs,
            )

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

    def plot_perimeter(
        self,
        inspect_pixels: bool = False,
        perimeter_border_normal_pixels: Optional[float] = None,
        with_midpoints: bool = False,
        manual_ax: Any = None,
        **plot_kwargs,
    ):
        if not manual_ax:
            fig, ax = plt.subplots()
        else:
            ax = manual_ax

        vertices = self.vertices_in_pixels if inspect_pixels else self.vertices_in_meters

        for index in range(len(vertices)):
            following_index = 0 if index + 1 == len(vertices) else index + 1

            corner_a = vertices[index]
            corner_b = vertices[following_index]
            ax.plot(
                *np.vstack((corner_a, corner_b)).T,
                # label=f"{self.label}{index}",     # Uncomment this when inspecting the sorting of edges
                **plot_kwargs,
            )

            if perimeter_border_normal_pixels is not None:
                perimeter = self.expand(perimeter_border_normal_pixels)
                border_a = perimeter[index]
                border_b = perimeter[following_index]
                ax.plot(
                    *np.vstack((border_a, border_b)).T,
                    **plot_kwargs,
                )

        if with_midpoints:
            for i, midpoint in enumerate(self.metric_graph.vertex_midpoints):
                ax.scatter(*midpoint.T, label=f"{self.label}{i}")

        if not manual_ax:
            generic_inspection_finalization(self.class_inspect_arg / f"{self.label}.jpg")

        return ax

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
