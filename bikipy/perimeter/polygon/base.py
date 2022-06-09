import copy
import json
from functools import cached_property
from logging import getLogger
from pathlib import Path
from typing import Any, ClassVar, Optional, Sequence, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pydantic import FilePath, validator, DirectoryPath
from pydantic_numpy import NDArray

from bikipy.perimeter.base import BasePerimeter
from bikipy.utils.io.makesense import image_name_to_point_from_makesense
from bikipy.utils.math.geometry import clockwise_sort_points, expand_bikipy_perimeter
from bikipy.utils.math.point_in_polygon import parallel_point_in_polygon
from bikipy.utils.math.vector import (
    normal_from_line_to_point,
    point_to_line_segment_distance,
)

logger = getLogger(__name__)


class PolygonPerimeter(BasePerimeter):
    corners: NDArray
    reference_point_coco_path: Optional[FilePath] = None
    reference_point_array: Optional[NDArray] = None
    inspect_image_path: Optional[FilePath] = None
    inspect_image_array: Optional[NDArray] = None
    feature_scale: Optional[NDArray] = None

    category: ClassVar[Optional[str]] = "perimeter"

    _polygon_order: ClassVar[Optional[int]] = None

    @validator("corners")
    def corners_polygon_order_validator(cls, value: NDArray):
        if cls._polygon_order and (n := len(value)) != int(cls._polygon_order):
            msg = (
                f"The polygon class is in the {cls._polygon_order}th order. However, "
                f"the current polygon is of the {n}th order"
            )
            raise ValueError(msg)
        return np.ascontiguousarray(clockwise_sort_points(value), dtype=np.float32)

    def __getitem__(self, item: int):
        return self.corners[item]

    def __repr__(self):
        return super().__repr__() + f"\n\tcorners={self.corners}"

    def expand(self, perimeter_border_normal_pixel_magnitude: Union[float, int]):
        """
        :param perimeter_border_normal_pixel_magnitude: The magnitude of the normal between
            the perimeter and the perimeter given in pixels
        :return:
        """
        if self._polygon_order == 4:
            from bikipy.perimeter import ParallelogramPerimeter

            border_obj = ParallelogramPerimeter(
                corners=expand_bikipy_perimeter(self, perimeter_border_normal_pixel_magnitude),
                inspect_image_array=self.inspect_image,
            )
        else:
            msg = f"Polygon order {self._polygon_order} is not supported"
            raise NotImplementedError(msg)

        return border_obj

    def closest_sides_to_coordinates(self, coordinates: Sequence):
        distance_sets = np.array(
            [
                point_to_line_segment_distance(coordinates, line_segment_pair)
                for line_segment_pair in self.line_segment_pairs
            ]
        ).T

        closest_boolean_index = np.argsort(distance_sets, axis=1) == 0
        closest_distance = distance_sets[closest_boolean_index]

        closest_index = np.where(closest_boolean_index)[1]

        closest_corner_start_point = np.zeros((closest_distance.shape[0], 2), dtype=np.float32)
        closest_corner_vectors = np.zeros((closest_distance.shape[0], 2), dtype=np.float32)
        for i in range(self.number_of_corners):
            closest_corner_start_point[closest_index == i] = self.corners[i]
            closest_corner_vectors[closest_index == i] = self.perimeter_corner_to_next_clockwise_corner_vectors[i]

        return closest_corner_start_point, closest_corner_vectors

    def closest_perimeter_points_to_coordinates(self, coordinates: Sequence):
        (
            closest_corner_start_point,
            closest_corner_vectors,
        ) = self.closest_sides_to_coordinates(coordinates)

        return normal_from_line_to_point(closest_corner_vectors, closest_corner_start_point, coordinates)

    def coordinate_confinement_boolean_index(self, coordinates: NDArray) -> NDArray:
        assert self.number_of_corners > 4
        return parallel_point_in_polygon(coordinates, self.corners)

    def change_reference(self, new_reference: Optional[NDArray], **new_inspect_image_kwargs):
        if np.all(self.reference_point == new_reference):
            logger.info("The provided reference_point is identical to the current")
            return self

        new_reference.astype(np.float64, copy=False)

        if not np.any(self.reference_point):
            new = self
            new.reference_point_array = new_reference
        else:
            new = copy.deepcopy(self)
            new.corners += new_reference - new.reference_point
            new.reference_point = new_reference

        return self._new_inspect_image(new, **new_inspect_image_kwargs)

    def plot_perimeter(
        self,
        perimeter_border_normal_pixel_magnitude: Union[float, int, None] = None,
        ax: Any = None,
        include_geometric_legend: bool = False,
        colormap: Any = None,
    ):
        if not ax:
            fig, ax = plt.subplots()

        legends = []
        for index in range(len(self.corners)):
            following_index = 0 if index + 1 == len(self.corners) else index + 1

            corner_a = self.corners[index]
            corner_b = self.corners[following_index]
            ax.plot(
                (corner_a[0], corner_b[0]),
                (corner_a[1], corner_b[1]),
                "o-",
                label=self.label,
                color=colormap,
            )
            ax.scatter(*self.edge_midpoints[index])

            if perimeter_border_normal_pixel_magnitude:
                perimeter = self.expand(perimeter_border_normal_pixel_magnitude)
                border_a = perimeter[index]
                border_b = perimeter[following_index]
                ax.plot(
                    (border_a[0], border_b[0]),
                    (border_a[1], border_b[1]),
                    "o-",
                    color=colormap,
                )

            if include_geometric_legend:
                legend = [
                    self._add_label_to_str(f"side {index}"),
                    self._add_label_to_str(f"midpoint {index}"),
                ]
                if perimeter_border_normal_pixel_magnitude:
                    legend.append(self._add_label_to_str(f"perimeter {index}"))

        plt.legend(legends, bbox_to_anchor=(1.04, 0.5), loc="center left")

        return ax

    @cached_property
    def number_of_corners(self):
        return len(self.corners)

    @cached_property
    def perimeter_corner_to_next_clockwise_corner_vectors(self):
        return np.diff(self.corners[::-1], prepend=[self.corners[0]], axis=0)[::-1]

    @cached_property
    def perimeter_lengths(self):
        return np.linalg.norm(self.perimeter_corner_to_next_clockwise_corner_vectors, axis=1)

    @cached_property
    def mean_length(self):
        return np.mean(self.perimeter_lengths)

    @cached_property
    def line_segment_pairs(self):
        pairs = [(self.corners[i], self.corners[i + 1]) for i in range(self.number_of_corners - 1)]
        pairs.append((self.corners[-1], self.corners[0]))
        return np.array(pairs)

    @cached_property
    def centroid(self):
        return np.mean(self.corners, axis=0)

    @cached_property
    def linked_corners(self):
        return np.append(self.corners, np.expand_dims(self.corners[0], 0), axis=0)

    @cached_property
    def edge_midpoints(self):
        return self.corners + np.diff(self.linked_corners, axis=0) / 2.0

    @cached_property
    def linked_polygon_edge_corner_pairs(self):
        return (
            *((i, i + 1) for i in range(self.number_of_corners - 1)),
            (self.number_of_corners - 1, 0),
        )

    @cached_property
    def y_flipped_edge_midpoints(self):
        # self.edge_midpoints.T[1].max()) is the maximum y value
        return np.array((0.0, self.corners_y_max)) - self.edge_midpoints

    @cached_property
    def y_flipped_edge_midpoint_scalars(self):
        return np.linalg.norm(self.y_flipped_edge_midpoints, axis=1)

    @cached_property
    def corners_y_max(self):
        return self.corners.T[1].max()

    @classmethod
    def init_polygon(cls, corners: NDArray, **kwargs):
        corners = np.asarray(corners)
        if (number_of_corners := corners.shape[0]) == 3:
            from bikipy.perimeter.polygon.triangular import TriangularPerimeter

            return TriangularPerimeter(corners=corners, **kwargs)
        elif number_of_corners == 4:
            from bikipy.perimeter.polygon.parallelogram import ParallelogramPerimeter

            return ParallelogramPerimeter(corners=corners, **kwargs)
        else:
            return cls(corners=corners, **kwargs)

    def _add_label_to_str(self, in_string):
        if self.label:
            return f"{self.label} {in_string}"
        if self.int_id:
            return f"{self.int_id} {in_string}"
        return in_string

    @classmethod
    def from_makesense_coco_polygon(
        cls,
        data_path: Any,
        image_root: Optional[DirectoryPath] = None,
        reference_point_csv_path: Optional[FilePath] = None,
        **perimeter_kwargs,
    ) -> dict:
        logger.debug("Generating PolygonPerimeter from makesense polygon data in coco format")

        with open(data_path, "rb") as in_json:
            coco = json.load(in_json)

        assert not image_root or (image_root := Path(image_root)).exists()

        # The coco annotations are not sorted with respect to the category IDs
        coco["annotations"] = sorted(coco["annotations"], key=lambda dictionary: dictionary["category_id"])

        # We don't need to do this, but better to be on the safe side
        coco["categories"] = sorted(coco["categories"], key=lambda dictionary: dictionary["id"])

        if reference_point_csv_path:
            image_name_to_reference_point = image_name_to_point_from_makesense(reference_point_csv_path)

        result = {}
        for annotation in coco["annotations"]:
            current_kwargs = {}
            image_name = coco["images"][annotation["image_id"] - 1]["file_name"]
            label = coco["categories"][annotation["category_id"] - 1]["name"]

            if image_root:
                assert not any(key in perimeter_kwargs for key in ("inspect_image_path", "inspect_image_array"))
                current_kwargs["inspect_image_path"] = image_root / image_name
            if reference_point_csv_path:
                current_kwargs["reference_point_array"] = image_name_to_reference_point[image_name]

            if image_name not in result:
                result[image_name] = {}

            result[image_name]["label"] = cls.init_polygon(
                _coco_polygon_annotation(annotation["segmentation"][0]),
                label=label,
                **current_kwargs,
                **perimeter_kwargs,
            )

        return image_name_to_point_from_makesense(result)

    @classmethod
    def from_makesense_csv_rectangle(
        cls,
        data_path: FilePath,
        image_root: Optional[DirectoryPath] = None,
        reference_point_csv_path: Optional[FilePath] = None,
        **perimeter_kwargs,
    ):
        csv_data = pd.read_csv(data_path, header=None, index_col=0)

        if reference_point_csv_path:
            image_name_to_reference_point = image_name_to_point_from_makesense(reference_point_csv_path)

        result = {}
        for label, row in csv_data.iterrows():
            image_name = row.values[4]

            start = np.array(row[:2]).astype(int)
            end = start + np.array(row[2:4]).astype(int)

            if image_name not in result:
                result[image_name] = {}

            result[image_name]["label"] = cls.init_polygon(
                np.array((start, (start[0], end[1]), end, (end[0], start[1]))),
                inspect_image_path=image_root / str(image_name) if image_root else None,
                label=label,
                reference_point_array=image_name_to_reference_point[image_name] if reference_point_csv_path else None,
                **perimeter_kwargs,
            )

        return image_name_to_point_from_makesense(result)


def _coco_polygon_annotation(flat_annotation_data: Sequence):
    return [(flat_annotation_data[i], flat_annotation_data[i + 1]) for i in range(0, len(flat_annotation_data) - 1, 2)]
