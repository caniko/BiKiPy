import json
from abc import ABC
from functools import cached_property
from logging import getLogger
from pathlib import Path
from typing import Any, ClassVar, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
from pydantic import DirectoryPath, FilePath, validator

from bikipy.core.typing import NDArrayFp64, NDArrayInt16
from bikipy.perimeter.base import (
    BasePerimeter,
    perimeter_set_from_image_name_to_perimeters,
)
from bikipy.utils.io.makesense import (
    image_name_to_point_from_makesense,
    read_makesense_rectangle,
)
from bikipy.utils.math.geometry import clockwise_sort_points, expand_rectangle
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
    inspect_image_path: Optional[FilePath]
    inspect_image_array: Optional[NDArrayFp64]
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
    def vertices_in_meters(self):
        return self.vertices_in_pixels * self.video.meters_per_pixel

    @cached_property
    def linked_vertices_in_meters(self):
        return np.append(self.vertices_in_meters, np.expand_dims(self.vertices_in_meters[0], 0), axis=0)

    @cached_property
    def clockwise_edge_unit_vectors(self):
        return unit_vector(np.diff(self.linked_vertices_in_meters, axis=0)[::-1])

    @cached_property
    def line_segment_pairs(self):
        return np.array(list(zip(self.linked_vertices_in_meters, self.linked_vertices_in_meters[1:])))

    def closest_point_on_edge_to_coordinates(self, coordinates: NDArrayFp64) -> NDArrayFp64:
        # Closest point on the index-respective edge along axis 0, and coordinates along 1.
        closest_edge_point_to_coordinates_matrix = np.array(
            [
                nearest_point_on_line_segment_to_coordinates(*line_segment_pair, coordinates)
                for line_segment_pair in self.line_segment_pairs
            ]
        )
        # Distance of the coordinate from the previous matrix
        distance_matrix = coordinates - closest_edge_point_to_coordinates_matrix

        closest_boolean_index = np.argsort(distance_matrix, axis=1) == 0

        return closest_edge_point_to_coordinates_matrix[closest_boolean_index].T

    def vector_to_closest_point_on_edge(self, coordinates: NDArrayFp64) -> NDArrayFp64:
        return unit_vector(self.closest_point_on_edge_to_coordinates(coordinates) - coordinates)

    def coordinate_confinement_boolean_index(self, coordinates: NDArrayFp64) -> NDArrayFp64:
        assert self.polygon_order >= 4, f"polygon_order <= 4, {self.polygon_order}"
        return parallel_point_in_polygon(coordinates, self.vertices_in_meters)

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

    @cached_property
    def centroid(self):
        return np.mean(self.vertices_in_meters, axis=0)

    @cached_property
    def vertex_neighbor_pairs(self):
        return (
            *((i, i + 1) for i in range(self.polygon_order - 1)),
            (self.polygon_order - 1, 0),
        )

    def plot_perimeter(
        self,
        inspect_pixels: bool = False,
        perimeter_border_normal_pixels: Optional[float] = None,
        ax: Any = None,
        include_geometric_legend: bool = False,
        colormap: Any = None,
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
            )

            if perimeter_border_normal_pixels:
                perimeter = self.expand(perimeter_border_normal_pixels)
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
                if perimeter_border_normal_pixels:
                    legend.append(self._add_label_to_str(f"perimeter {index}"))

        plt.legend(legends, bbox_to_anchor=(1.04, 0.5), loc="center left")

        return ax

    @classmethod
    def init_polygon(cls, vertices_in_meters: NDArrayFp64, **kwargs):
        vertices_in_meters = np.asarray(vertices_in_meters)
        polygon_order = vertices_in_meters.shape[0]
        if polygon_order == 3:
            from bikipy.perimeter.polygon.triangular import TriangularPerimeter

            return TriangularPerimeter(vertices_in_pixels=vertices_in_meters, **kwargs)
        elif polygon_order == 4:
            from bikipy.perimeter.polygon.rectangle import RectanglePerimeter

            return RectanglePerimeter(vertices_in_pixels=vertices_in_meters, **kwargs)
        else:
            msg = f"Polygon of the {polygon_order}nt order is not supported"
            raise NotImplementedError(msg)

    def _add_label_to_str(self, in_string):
        if self.label:
            return f"{self.label} {in_string}"
        if self.int_id:
            return f"{self.int_id} {in_string}"
        return in_string

    # Class methods for makesense integration ==========================================

    @classmethod
    def from_makesense_coco_polygon(
        cls,
        data_path: Any,
        image_root: Optional[DirectoryPath],
        reference_point_csv_path: Optional[FilePath],
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
        logger.debug("Generating PolygonPerimeter from makesense polygon data in coco format")

        csv_data = read_makesense_rectangle(data_path)

        if reference_point_csv_path:
            image_name_to_reference_point = image_name_to_point_from_makesense(reference_point_csv_path)

        result = {}
        for label, row in csv_data.iterrows():
            start = np.array(row[:2], dtype=int)
            end = start + np.array(row[2:4], dtype=int)

            image_name = row["image_name"]
            if image_name not in result:
                result[image_name] = {}

            result[image_name][label] = cls.init_polygon(
                np.array((start, (start[0], end[1]), end, (end[0], start[1]))),
                inspect_image_path=image_root / str(image_name) if image_root else None,
                label=label,
                reference_point_array=image_name_to_reference_point[image_name] if reference_point_csv_path else None,
                manual_recording_resolution=np.array((row["x_res"], row["y_res"]), dtype=float),
                **perimeter_kwargs,
            )

        return perimeter_set_from_image_name_to_perimeters(result)


def _coco_polygon_annotation(flat_annotation_data: Sequence):
    return [(flat_annotation_data[i], flat_annotation_data[i + 1]) for i in range(0, len(flat_annotation_data) - 1, 2)]
