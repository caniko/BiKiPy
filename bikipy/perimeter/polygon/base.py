from abc import ABC
from functools import cached_property
from logging import getLogger
from typing import Any, ClassVar, Optional

import matplotlib.pyplot as plt
import numpy as np
from pydantic import FilePath, validator

from bikipy.core.typing import NDArrayFp64, NDArrayInt16, NDArrayBool
from bikipy.core.video import convert_meters_to_pixels
from bikipy.perimeter.base import BasePerimeter
from bikipy.perimeter.radial.circle import CirclePerimeter
from bikipy.utils.collection_utils import evenly_spaced_indices
from bikipy.utils.math.geometry import clockwise_sort_points
from bikipy.utils.math.point_in_polygon import parallel_point_in_polygon
from bikipy.utils.math.vector import (
    nearest_point_on_line_segment_to_coordinates,
    unit_vector,
)

logger = getLogger(__name__)


class PolygonPerimeter(BasePerimeter, ABC):
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
    def centroid(self):
        return np.mean(self.vertices_in_meters, axis=0)

    @cached_property
    def vertices_in_meters(self):
        return self.vertices_in_pixels * self.video.meters_per_pixel

    @cached_property
    def linked_vertices_in_meters(self):
        return np.append(self.vertices_in_meters, np.expand_dims(self.vertices_in_meters[0], 0), axis=0)

    @cached_property
    def line_segment_pairs(self):
        return np.array(list(zip(self.linked_vertices_in_meters, self.linked_vertices_in_meters[1:])))

    @cached_property
    def edge_lengths(self):
        return np.linalg.norm(np.diff(self.line_segment_pairs, axis=0), axis=1)

    @cached_property
    def equilateral(self) -> bool:
        return np.all(
            np.apply_along_axis(np.isclose, 0, self.edge_lengths[0], self.edge_lengths[1:], atol=1.0e-4), axis=1
        )

    @cached_property
    def circle(self):
        return CirclePerimeter(
            center_pixels=convert_meters_to_pixels(self.centroid, self.video),
            radius_meters=np.mean(self.edge_lengths),
            manual_video=self.video,
        )

    def closest_point_on_edge_to_coordinates(self, coordinates: NDArrayFp64, inspect: bool = False) -> NDArrayFp64:
        # Closest point on the index-respective edge along axis 0, and coordinates along 1.
        closest_edge_point_to_coordinates_matrix = np.array(
            [
                nearest_point_on_line_segment_to_coordinates(*line_segment_pair, coordinates)
                for line_segment_pair in self.line_segment_pairs
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

            for i in evenly_spaced_indices(coordinates, 9):
                fig, ax = plt.subplots()
                to_skip = []
                for y, point in enumerate(indexable_t[i].T):
                    if y in to_skip:
                        continue
                    duplicates_boolean_indices = np.all(
                        np.apply_along_axis(np.isclose, 0, point, indexable_t[i].T, atol=1.0e-4), axis=1
                    )
                    sort_indices = ", ".join(argsorted_distance.T[i][duplicates_boolean_indices].astype(str))
                    ax.scatter(*point, label=sort_indices)

                    to_skip.extend(np.where(duplicates_boolean_indices)[0].tolist())

                ax.scatter(*coordinates[i], label="coordinate")
                plt.legend()
                plt.tight_layout()
                plt.show()

        return result

    def vector_to_closest_point_on_edge(self, coordinates: NDArrayFp64) -> NDArrayFp64:
        return unit_vector(self.closest_point_on_edge_to_coordinates(coordinates) - coordinates)

    def coordinate_confinement_boolean_index(self, coordinates: NDArrayFp64) -> NDArrayBool:
        return parallel_point_in_polygon(coordinates, self.vertices_in_meters)

    def gaze_direction_filter(
        self,
        gaze_travel_direction_point: NDArrayFp64,
        gaze_start_point: NDArrayFp64,
        max_radians: float,
        inspect: bool = False,
    ):
        gaze_vector = gaze_travel_direction_point - gaze_start_point

        closest_points_on_edges = self.closest_point_on_edge_to_coordinates(gaze_travel_direction_point)
        vector_to_closest_point_on_edge = self.vector_to_closest_point_on_edge(gaze_travel_direction_point)

        direction_point_is_closer_than_start_point = np.linalg.norm(
            closest_points_on_edges - gaze_travel_direction_point, axis=1
        ) < np.linalg.norm(closest_points_on_edges - gaze_start_point, axis=1)

    def change_reference(self, new_reference: NDArrayFp64, **new_inspect_image_kwargs):
        if self.reference_point is None:
            return self

        if np.all(self.reference_point == new_reference):
            logger.warning("The provided reference_point is identical to the current")
            return self

        if self.reference_point is not None and np.any(self.reference_point):
            return self.__class__(
                vertices_in_pixels=self.vertices_in_pixels + new_reference - self.reference_point,
                manual_video=self.video,
            )

        return self

    def plot_perimeter(
        self,
        inspect_pixels: bool = False,
        perimeter_border_normal_pixels: Optional[float] = None,
        ax: Any = None,
        include_geometric_legend: bool = False,
        colormap: Any = None,
        **plot_kwargs,
    ):
        if not ax:
            fig, ax = plt.subplots()

        vertices_in_meters = self.vertices_in_pixels if inspect_pixels else self.vertices_in_meters

        legends = []
        for index in range(len(vertices_in_meters)):
            following_index = 0 if index + 1 == len(vertices_in_meters) else index + 1

            corner_a = vertices_in_meters[index]
            corner_b = vertices_in_meters[following_index]
            ax.plot(
                (corner_a[0], corner_b[0]),
                (corner_a[1], corner_b[1]),
                "o-",
                label=self.label,
                color=colormap,
                **plot_kwargs,
            )

            if perimeter_border_normal_pixels is not None:
                perimeter = self.expand(perimeter_border_normal_pixels)
                border_a = perimeter[index]
                border_b = perimeter[following_index]
                ax.plot((border_a[0], border_b[0]), (border_a[1], border_b[1]), "o-", color=colormap, **plot_kwargs)

            if include_geometric_legend:
                legend = [
                    self._add_label_to_str(f"side {index}"),
                    self._add_label_to_str(f"midpoint {index}"),
                ]
                if perimeter_border_normal_pixels:
                    legend.append(self._add_label_to_str(f"perimeter {index}"))

        plt.legend(legends, bbox_to_anchor=(1.04, 0.5), loc="center left")

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
