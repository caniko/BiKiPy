from abc import ABC
from functools import cached_property
from logging import getLogger
from typing import Any, ClassVar, Optional

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
from bikipy.utils.math.geometry import clockwise_sort_points
from bikipy.utils.math.point_in_polygon import parallel_point_in_polygon
from bikipy.utils.math.vector import (
    nearest_point_on_line_segment_to_coordinates,
    unit_vector,
    ray_and_line_segment_intersection,
    rotate_vectors_with_angle,
)

logger = getLogger(__name__)


class PolygonPerimeter(BaseSinglePerimeter, ABC):
    vertices_in_pixels: NDArrayFp64
    reference_point_coco_path: Optional[FilePath]
    reference_point_array: Optional[NDArrayInt16]
    feature_scale: Optional[NDArrayFp64]

    category: ClassVar[Optional[str]] = "perimeter"

    polygon_order: ClassVar[Optional[int]]

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
        return np.ascontiguousarray(clockwise_sort_points(value))

    def __getitem__(self, item: int):
        return self.vertices_in_meters[item]

    def __repr__(self):
        return super().__repr__() + f"\n\tvertices_in_pixels={self.vertices_in_meters}"

    @cached_property
    def centroid(self) -> NDArrayFp64:
        return np.mean(self.vertices_in_meters, axis=0)

    @cached_property
    def vertices_in_meters(self) -> NDArrayFp64:
        return self.vertices_in_pixels * self.video.meters_per_pixel

    @cached_property
    def linked_vertices_meters(self) -> NDArrayFp64:
        return np.append(
            self.vertices_in_meters,
            np.expand_dims(self.vertices_in_meters[0], 0),
            axis=0,
        )

    @cached_property
    def line_segment_points_meters(self) -> NDArrayFp64:
        return np.array(list(zip(self.linked_vertices_meters, self.linked_vertices_meters[1:])))

    @cached_property
    def line_segment_midpoints_meters(self) -> NDArrayFp64:
        return (
            self.linked_vertices_meters[1:]
            + np.diff(self.line_segment_points_meters, axis=1).transpose(1, 0, 2)[0] / 2.0
        )

    @cached_property
    def edge_lengths_meters(self):
        return np.linalg.norm(np.diff(self.line_segment_points_meters, axis=0), axis=1)

    @cached_property
    def linked_vertices_pixels(self) -> NDArrayFp64:
        return np.append(
            self.vertices_in_pixels,
            np.expand_dims(self.vertices_in_pixels[0], 0),
            axis=0,
        )

    @cached_property
    def line_segment_points_pixels(self) -> NDArrayFp64:
        return np.array(list(zip(self.linked_vertices_pixels, self.linked_vertices_pixels[1:])))

    @cached_property
    def line_segment_midpoints_pixels(self) -> NDArrayFp64:
        return self.vertices_in_pixels - np.diff(self.line_segment_points_pixels, axis=1).transpose(1, 0, 2)[0] / 2.0

    @cached_property
    def edge_lengths_pixels(self) -> NDArrayFp64:
        return np.linalg.norm(np.diff(self.line_segment_points_pixels, axis=0), axis=1)

    @cached_property
    def equilateral(self) -> bool:
        return np.all(
            np.apply_along_axis(np.isclose, 0, self.edge_lengths_meters[0], self.edge_lengths_meters[1:], atol=1.0e-4),
            axis=1,
        )

    @cached_property
    def circle(self):
        return CirclePerimeter(
            center_pixels=convert_meters_to_pixels(self.centroid, self.video),
            radius_meters=np.mean(self.edge_lengths_meters),
            manual_video=self.video,
        )

    def closest_point_on_edge_to_coordinates(self, coordinates: NDArrayFp64, inspect: bool = False) -> NDArrayFp64:
        # Closest point on the index-respective edge along axis 0, and coordinates along 1.
        closest_edge_point_to_coordinates_matrix = np.array(
            [
                nearest_point_on_line_segment_to_coordinates(*line_segment_pair, coordinates)
                for line_segment_pair in self.line_segment_points_meters
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

    def coordinate_confinement_boolean_index(self, coordinates: NDArrayFp64) -> NDArrayBool:
        return parallel_point_in_polygon(coordinates, self.vertices_in_meters)

    def ray_intersects_on_polygon(
        self,
        ray_origins: NDArrayFp64,
        ray_directions: NDArrayFp64,
        return_points: bool = False,
    ) -> NDArray:
        result = np.array(
            [
                ray_and_line_segment_intersection(ray_origins, ray_directions, *line_segment_pair, return_points)
                for line_segment_pair in self.line_segment_points_meters
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
        manual_inspect: bool = True,
        **inspect_kwargs,
    ) -> NDArrayBool:
        """
        Determine if the object is within the gaze cone

        This problem is called the "in line of sight" (ilos) problem, and is non-trivial. This is not the best solution
        in terms of speed for our application; nevertheless, it is quite robust and had the lowest implementation time.
        The solution below does the following:
            1.

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

        if self.inspect or manual_inspect:
            from bikipy.feature.attention.gaze import gaze_inspection_plot

            gaze_inspection_plot(
                self,
                result,
                gaze_vectors,
                gaze_travel_direction_point,
                **inspect_kwargs,
            )

        return result

    def change_reference(self, new_reference: NDArrayFp64, makesense_image_name: Optional[str] = None):
        if self.reference_point is None:
            msg = "Reference without defining a reference for the source perimeter object is disallowed"
            raise AttributeError(msg)

        if np.all(self.reference_point == new_reference):
            logger.warning("The provided reference_point is identical to the current")
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
        inspection_ax: Any = None,
        **plot_kwargs,
    ):
        if not inspection_ax:
            fig, ax = plt.subplots()
        else:
            ax = inspection_ax

        vertices_in_meters = self.vertices_in_pixels if inspect_pixels else self.vertices_in_meters

        for index in range(len(vertices_in_meters)):
            following_index = 0 if index + 1 == len(vertices_in_meters) else index + 1

            corner_a = vertices_in_meters[index]
            corner_b = vertices_in_meters[following_index]
            ax.plot(
                *np.vstack((corner_a, corner_b)).T,
                label=f"{self.label}{index}",
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
            for i, midpoint in enumerate(self.line_segment_midpoints_meters):
                ax.scatter(*midpoint.T, label=f"{self.label}{i}")

        if not inspection_ax:
            plt.legend()
            plt.show()
        elif self.inspect_directory:
            plt.savefig(self.class_inspect_directory / f"{self.label}.jpg")

        return ax

    def _add_label_to_str(self, in_string):
        if self.label:
            return f"{self.label} {in_string}"
        if self.int_id:
            return f"{self.int_id} {in_string}"
        return in_string


def init_polygon(vertices_in_meters: NDArrayFp64, **kwargs):
    match vertices_in_meters.shape[0]:  # polygon_order
        case 3:
            from bikipy.perimeter.polygon.triangular import TriangularPerimeter

            return TriangularPerimeter(vertices_in_pixels=vertices_in_meters, **kwargs)
        case 4:
            from bikipy.perimeter.polygon.rectangle import RectanglePerimeter

            return RectanglePerimeter(vertices_in_pixels=vertices_in_meters, **kwargs)
        case _:
            return PolygonPerimeter(vertices_in_pixels=vertices_in_meters, **kwargs)
