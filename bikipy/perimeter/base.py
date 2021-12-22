import copy
import json
import statistics
from functools import cached_property, lru_cache
from logging import getLogger
from pathlib import Path, PurePath
from typing import Any, ClassVar, Optional, Sequence, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.typing import NDArray as NpNDArray
from pydantic import DirectoryPath, FilePath, validator
from shapely.geometry import Point, Polygon

from bikipy._base_class import BikipyBase
from bikipy.math.geometry import clockwise_sort_points, expand_bikipy_perimeter
from bikipy.math.vector import point_to_line_segment_distance
from bikipy.utils.misc import (
    get_reference_point_from_array,
    read_image,
    read_makesense_point_csv,
    to_tuple,
)
from bikipy.utils.typing import NDArray, OptionalPathTyping

logger = getLogger(__name__)


class BasePerimeter(BikipyBase):
    reference_point_coco_path: Optional[FilePath] = None
    reference_point_array: Optional[NDArray] = None
    inspect_image: Optional[NDArray] = None
    image_name: Optional[str] = None

    category: ClassVar[Optional[str]] = "perimeter"

    @validator("inspect_image", pre=True)
    def make_sure_image_is_loaded(cls, value):
        if isinstance(value, (str, PurePath)):
            if not (path := Path(value)).exists():
                msg = "The provided path to image for inspection, doesn not exist"
                raise ValueError(msg)
            return read_image(path)
        elif np.any(value) or value is None:
            return value
        else:
            msg = (
                "inspect_image:P The provided object is not a numpy array; it is not "
                "an image."
            )
            raise ValueError(msg)

    @property
    def reference_point(self):
        if self.reference_point_array is None and not self.reference_point_coco_path:
            return None
        return (
            _reference_point_from_coco_path(self.reference_point_coco_path)
            if self.reference_point_array is None
            else self.reference_point_array
        )

    @reference_point.setter
    def reference_point(self, value):
        self.reference_point_array = np.asarray(value)

    def plot(self, ax: Any = None, points: Optional[Sequence] = None):
        """
        Plot the perimeter using matplotlib. Optionally, plot points alongside the perimeter

        Parameters
        ----------
        ax
            Axes object that the plot will be saved in. A new instance of Axes will be used
            if object returns False.
        points
            Sequence of 2D coordinates that will be plotted alongside the perimeter

        Returns
        -------
        Axes object with plots
        """
        if not ax:
            fig, ax = plt.subplots()

        if self.inspect_image is not None:
            ax.imshow(self.inspect_image)

        if points is not None:
            points = np.asarray(points)

            histogram, _x_edges, _y_edges = np.histogram2d(
                *points[np.logical_and(*np.isfinite(points).T)].T, bins=60
            )
            ax.imshow(histogram.T, interpolation="sinc")
            ax.plot(*points.T, ".r-")

        ax.set_title(self.best_id)

        return ax

    @staticmethod
    def _get_coco_array_from_path_or_array(
        coco_path: Optional[FilePath] = None,
        coco_array: Optional[np.ndarray] = None,
    ):
        msg = "coco_path and coco_array are defined mutually exclusive"
        if coco_path and np.any(coco_array):
            raise ValueError(msg)

        if coco_path:
            result = read_makesense_point_csv(coco_path)
        elif np.any(coco_array):
            result = coco_array
        else:
            raise ValueError(msg)

        assert np.any(result)
        return result


class Perimeter(BasePerimeter):
    corners: np.ndarray
    feature_scale: Optional[NDArray] = None

    _polygon_order: ClassVar[Optional[int]] = None

    @validator("corners")
    def corners_polygon_order_validator(cls, value: NpNDArray):
        if cls._polygon_order and (n := len(value)) != int(cls._polygon_order):
            msg = (
                f"The polygon class is in the {cls._polygon_order}th order. However, "
                f"the current polygon is of the {n}th order"
            )
            raise ValueError(msg)
        return clockwise_sort_points(value)

    def __getitem__(self, item: int):
        return self.corners[item]

    def __repr__(self):
        return super().__repr__() + f"\n\tcorners={self.corners}"

    def border(self, perimeter_border_normal_pixel_magnitude: Union[float, int]):
        """
        :param perimeter_border_normal_pixel_magnitude: The magnitude of the normal between
            the perimeter and the border given in pixels
        :return:
        """
        if self._polygon_order == 4:
            from bikipy.perimeter import ParallelogramPerimeter

            border_obj = ParallelogramPerimeter(
                corners=expand_bikipy_perimeter(
                    self, perimeter_border_normal_pixel_magnitude
                ),
                inspect_image=self.inspect_image,
            )
        else:
            msg = f"Polygon order {self._polygon_order} is not supported"
            raise NotImplementedError(msg)

        return border_obj

    def closest_sides_to_points(self, points: Sequence):
        distance_sets = np.array(
            [
                point_to_line_segment_distance(points, line_segment_pair)
                for line_segment_pair in self.line_segment_pairs
            ]
        ).T

        closest_boolean_index = np.argsort(distance_sets, axis=1) == 0
        closest_distance = distance_sets[closest_boolean_index]

        closest_index = np.where(closest_boolean_index)[1]
        closest_vectors = np.zeros((closest_distance.shape[0], 2), dtype=np.float32)
        for i in range(self.number_of_corners):
            closest_vectors[closest_index == i] = self.perimeter_vectors[i]

        return closest_distance, closest_vectors

    def confined_coordinates(
        self, coordinates: Sequence, inspect: bool = False, ax: Any = None
    ):
        """
        self.confined_coordinates to fetch confined coordinates within
        the respective perimeter

        :param coordinates: Coordinates that will have their confinement tested
        :param inspect: If True, plot the confined coordinates
        :type coordinates: np.ndarray
        :type inspect: bool
        :return: Coordinates cointain
        :rtype: np.ndarray
        """
        coordinates = np.asarray(coordinates)
        coordinate_confinement_boolean_index = coordinates[
            self.coordinate_confinement_boolean_index(coordinates)
        ]
        if inspect or ax:
            if not ax:
                ax = self.plot_self()
            ax.scatter(
                coordinate_confinement_boolean_index.T[0],
                coordinate_confinement_boolean_index.T[1],
                marker="x",
            )
            ax.set_tittle("Confined coordinates")
            plt.show()

        return coordinate_confinement_boolean_index

    def coordinate_confinement_boolean_index(self, coordinates: Sequence) -> np.ndarray:
        assert self.number_of_corners > 4

        polygon = Polygon(self.corners)
        return np.array(
            [polygon.contains(Point(coordinate)) for coordinate in coordinates]
        )

    @classmethod
    def detect_sequential_border_presence(
        cls,
        coordinates: Sequence[Sequence[float]],
        superior_poly_border_instances: Optional[Sequence],
        inferior_poly_border_instances: Optional[Sequence] = None,
        clean_outliers: bool = True,
    ):
        """
        Define sequential perimeter confinements of coordinates

        Parameters
        ----------
        coordinates: Sequence
            Coordinates that will have their confinement tested

        superior_poly_border_instances: Sequence
            Perimeter instances that will have the highest priority
            in case of overlap with respect to confinement

        inferior_poly_border_instances: Sequence
            Perimeter instances that will have the lowest priority
            in case of overlap with respect to confinement

        clean_outliers
            Clear elements that aren't confined to any of the given border_corners
            as a final action before returning the sequential perimeter presence

        Returns
        -------
        np.ndarray that stores the sequential perimeter presence across frames
        """

        coordinates = np.asarray(coordinates)

        border_sequence = (
            (*inferior_poly_border_instances, *superior_poly_border_instances)
            if inferior_poly_border_instances
            else superior_poly_border_instances
        )
        presence = np.zeros(
            coordinates.shape[0],
            dtype=np.int8 if len(border_sequence) <= 255 else np.int16,
        )

        overlap_locations = {}
        for border in border_sequence:
            confined_coord_booleans_index = border.coordinate_confinement_boolean_index(
                coordinates
            )

            if presence[confined_coord_booleans_index].any():
                overlap_locations[border.label] = np.flatnonzero(
                    presence[confined_coord_booleans_index]
                )
                presence[overlap_locations[border.label]] = 0
                logger.info(
                    f"BasePerimeter {border.label} has coordinate overlap with "
                    f"other border_corners, {overlap_locations[border.label].size}"
                )

            presence[confined_coord_booleans_index] = border.int_id

        valid_indices = np.nonzero(presence)
        if clean_outliers:
            presence = presence[valid_indices]

        boolean_array = np.full(coordinates.shape[0], False, dtype=np.bool)
        boolean_array[valid_indices] = True

        return presence, valid_indices, boolean_array

    def change_reference(
        self,
        new_reference: Optional[np.ndarray],
        new_inspect_image: Optional[np.ndarray] = None,
        new_inspect_image_path: Optional[FilePath] = None,
    ):
        assert np.any(self.reference_point)
        if np.all(self.reference_point == new_reference):
            logger.info("The provided reference_point is identical to the current")
            return self

        new_reference.astype(np.float64, copy=False)

        new = copy.deepcopy(self)
        new.corners += new_reference - new.reference_point
        new.reference_point = new_reference

        if new_inspect_image_path:
            if not (new_inspect_image_path := Path(new_inspect_image_path)).exists():
                msg = (
                    f"new_inspect_image_path, {new_inspect_image_path}, does not exist"
                )
                raise AttributeError(msg)
            new.inspect_image = new_inspect_image_path
        elif np.any(new_inspect_image):
            new.inspect_image = new_inspect_image
        else:
            new.inspect_image = None

        return new

    def change_reference_with_coco(
        self,
        coco_path: Optional[FilePath] = None,
        coco_array: Optional[np.ndarray] = None,
        **kwargs,
    ):
        coco_array = self._get_coco_array_from_path_or_array(coco_path, coco_array)

        if len(coco_array) != 1:
            msg = (
                "The coco array includes more than one annotation. "
                "Please use change_reference_with_coco_with_plural_references()"
            )
            raise ValueError(msg)

        return self.change_reference(
            get_reference_point_from_array(coco_array), **kwargs
        )

    def change_reference_with_coco_with_plural_references(
        self,
        coco_path: Optional[FilePath] = None,
        coco_array: Optional[np.ndarray] = None,
        image_root: Optional[DirectoryPath] = None,
        map_to_image_names: bool = True
    ):
        def _change_reference_loop_func(reference_point, img_name):
            return self.change_reference(
                reference_point,
                new_inspect_image_path=image_root / img_name if image_root else None,
            )

        coco_array = self._get_coco_array_from_path_or_array(coco_path, coco_array)

        img_name_vs_reference_points = {
            row[3]: get_reference_point_from_array(row) for row in coco_array
        }
        if not np.any(self.reference_point):
            msg = "The reference polygon has no reference point"
            raise ValueError(msg)

        if map_to_image_names:
            return {
                img_name: _change_reference_loop_func(reference_point, img_name)
                for img_name, reference_point in img_name_vs_reference_points.items()
            }
        return [
            _change_reference_loop_func(reference_point, img_name)
            for img_name, reference_point in img_name_vs_reference_points.items()
        ]

    def plot_self(
        self,
        plot_kwargs: Optional[dict] = None,
        perimeter_plot_kwargs: Optional[dict] = None,
    ):
        """
        Plot the corners defined in the object, along with

        Returns
        -------
        matplotlib Axes object with the plot
        """
        plot_kwargs = plot_kwargs or {}
        ax = super().plot(**plot_kwargs)

        perimeter_plot_kwargs = perimeter_plot_kwargs or {}
        self.plot_perimeter(ax=ax, **perimeter_plot_kwargs)
        return ax

    @classmethod
    def plot_perimeters(
        cls,
        perimeters: Sequence,
        ax: Any = None,
        inspect_image: Any = None,
        perimeter_plot_kwargs: Optional[dict] = None,
    ):
        if not ax:
            _fig, ax = plt.subplots()

        if inspect_image is None:
            for i, perimeter in enumerate(perimeters):
                if isinstance(perimeter.inspect_image, np.ndarray):
                    potential_inspect_image = perimeter.inspect_image
                    if i == len(perimeters) - 1 or all(
                        perimeter.inspect_image is None
                        or np.all(potential_inspect_image == perimeter.inspect_image)
                        for perimeter in perimeters[i + 1 :]
                    ):
                        """
                        Old premature optimisation, DON'T DO THIS AGAIN.
                        Use the found image if and only if it is identical
                        to other inspect_images in the rest of the perimeter objects
                        """
                        inspect_image = potential_inspect_image
                    break

        if inspect_image is not None:
            ax.imshow(read_image(inspect_image), cmap="gray", vmin=0, vmax=255)

        perimeter_plot_kwargs = perimeter_plot_kwargs or {}
        for perimeter in perimeters:
            perimeter.plot_perimeter(ax=ax, **perimeter_plot_kwargs)

        return ax

    def plot_perimeter(
        self,
        perimeter_border_normal_pixel_magnitude: Union[float, int, None] = None,
        ax: Any = None,
        include_geometric_legend: bool = False,
        color: Any = None,
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
                color=color,
            )
            ax.scatter(*self.edge_midpoints[index])

            if perimeter_border_normal_pixel_magnitude:
                border = self.border(perimeter_border_normal_pixel_magnitude)
                border_a = border[index]
                border_b = border[following_index]
                ax.plot(
                    (border_a[0], border_b[0]),
                    (border_a[1], border_b[1]),
                    "o-",
                    color=color,
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
    def perimeter_vectors(self):
        return np.diff(self.corners[::-1], prepend=[self.corners[0]], axis=0)[::-1]

    @cached_property
    def perimeter_lengths(self):
        return np.linalg.norm(self.perimeter_vectors, axis=1)

    @cached_property
    def mean_length(self):
        return np.mean(self.perimeter_lengths)

    @cached_property
    def line_segment_pairs(self):
        pairs = [
            (self.corners[i], self.corners[i + 1])
            for i in range(self.number_of_corners - 1)
        ]
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
    def init_polygon(cls, corners: Sequence, **kwargs):
        corners = np.asarray(corners)
        if (number_of_corners := corners.shape[0]) == 3:
            from bikipy.perimeter.triangular import TriangularPerimeter

            return TriangularPerimeter(corners=corners, **kwargs)
        elif number_of_corners == 4:
            from bikipy.perimeter.parallelogram.classes import ParallelogramPerimeter

            return ParallelogramPerimeter(corners=corners, **kwargs)
        else:
            return cls(corners=corners, **kwargs)

    @classmethod
    def from_polygon_coco(
        cls,
        coco_path: Any,
        image_root: Optional[DirectoryPath] = None,
        single_obj_return: bool = False,
        **perimeter_kwargs,
    ) -> Union[dict, BasePerimeter]:
        def get_inspect_image_name(image_id: int):
            return coco["images"][image_id - 1]["file_name"]

        def get_inspect_image_path(image_id: int):
            return (
                image_root / coco["images"][image_id - 1]["file_name"]
                if image_root
                else None
            )

        def get_semantic_label(category_id: int):
            return coco["categories"][category_id - 1]["name"].lower()

        logger.debug("Generating Perimeter from coco data")

        with open(coco_path, "rb") as in_json:
            coco = json.load(in_json)

        assert not image_root or (image_root := Path(image_root)).exists()

        # The coco annotations are not sorted with respect to the category IDs
        coco["annotations"] = sorted(
            coco["annotations"], key=lambda dictionary: dictionary["category_id"]
        )

        # We don't need to do this, but better to be on the safe side
        coco["categories"] = sorted(
            coco["categories"], key=lambda dictionary: dictionary["id"]
        )

        semantic_label_vs_polygon = {
            get_semantic_label(annotation["category_id"]): cls.init_polygon(
                _coco_polygon_annotation(annotation["segmentation"][0]),
                inspect_image=get_inspect_image_path(annotation["image_id"]),
                image_name=get_inspect_image_name(annotation["image_id"]),
                label=coco["categories"][annotation["category_id"] - 1]["name"],
                **perimeter_kwargs,
            )
            for annotation in coco["annotations"]
        }

        if single_obj_return:
            assert (
                len(semantic_label_vs_polygon) == 1
            ), f"More than one item in coco set, {len(semantic_label_vs_polygon)}"
            return semantic_label_vs_polygon.popitem()[1]

        return semantic_label_vs_polygon

    @classmethod
    def from_makesense_ai(
        cls, metadata_path: FilePath, image_root: DirectoryPath, **perimeter_kwargs
    ):
        if metadata_path.suffix == ".csv":  # rectangle object
            csv_data = pd.read_csv(metadata_path, header=None, index_col=0)
            for label, row in csv_data.iterrows():
                start = np.array(row[:2])
                end = start + np.array(row[2:4])
                cls.init_polygon(
                    (start, (start[0], end[1]), end, (end[0], start[1])),
                    inspect_image=get_inspect_image_path(annotation["image_id"]),
                    image_name=row[4],
                    label=label,
                    **perimeter_kwargs,
                )

    def _add_label_to_str(self, in_string):
        if self.label:
            return f"{self.label} {in_string}"
        if self.int_id:
            return f"{self.int_id} {in_string}"
        return in_string


class PerimeterSet(BasePerimeter):
    perimeters: Union[tuple, dict]
    restricted_perimeters: Union[tuple, dict, None] = None

    def __init__(self, **data):
        super().__init__(**data)
        if self.reference_point is not None:
            for perimeter in self.perimeters:
                perimeter.reference_point_array = self.reference_point

            if self.restricted_perimeters:
                for perimeter in self.restricted_perimeters:
                    perimeter.reference_point_array = self.reference_point

    def __getitem__(self, item: Union[str, int]):
        for perimeter in self._all_perimeters:
            if perimeter.label == item or perimeter.int_id == item:
                return perimeter
        raise KeyError(f"Item was not found, {item}")

    @cached_property
    def group(self):
        grouped = {}
        for perimeter in self._all_perimeters:
            if (label := perimeter.group_label) not in grouped:
                grouped[label] = [perimeter]
            else:
                grouped[label].append(perimeter)

        # Groups with one perimeter member should be the value of the respective key
        for label, perimeters in grouped.items():
            if len(perimeters) == 1:
                grouped[label] = perimeters[0]
            else:
                grouped[label] = tuple(perimeters)

        return grouped

    @cached_property
    def centroid(self):
        """
        :return: The mean of all perimeter centroids in the set
        """
        return np.mean([perimeter.centroid for perimeter in self.perimeters], axis=0)

    def discrete_confined_coordinates(
        self, coordinates: Sequence, inspect: bool = False
    ):
        ax = self.plot() if inspect else None
        result = {}
        for i, perimeter in enumerate(self._all_perimeters):
            confined_coordinates = perimeter.confined_coordinates(coordinates, ax=ax)
            result[perimeter.best_id or i] = confined_coordinates
        if inspect:
            plt.show()
        return result

    def combined_confined_coordinates(self, coordinates: Sequence):
        present = np.any(
            [
                perimeter.coordinate_confinement_boolean_index(coordinates)
                for perimeter in self.perimeters
            ]
        )
        if self.restricted_perimeters:
            present = present & ~np.any(
                [
                    perimeter.coordinate_confinement_boolean_index(coordinates)
                    for perimeter in self.restricted_perimeters
                ]
            )
        return present

    def change_reference_with_coco(
        self,
        coco_path: Optional[FilePath] = None,
        coco_array: Optional[np.ndarray] = None,
        **kwargs,
    ):
        coco_array = self._get_coco_array_from_path_or_array(coco_path, coco_array)

        if len(coco_array) != 1:
            msg = (
                "The coco array includes more than one annotation. "
                "Please use change_reference_with_coco_with_plural_references()"
            )
            raise ValueError(msg)

        return self.__class__(
            perimeters=[
                perimeter.change_reference_with_coco(coco_array)
                for perimeter in self.perimeters
            ],
            restricted_perimeters=[
                perimeter.change_reference(coco_array)
                for perimeter in self.restricted_perimeters
            ]
            if self.restricted_perimeters
            else None,
            **kwargs,
        )

    def change_reference_with_coco_with_plural_references(
        self,
        coco_path: Optional[FilePath] = None,
        coco_array: Optional[np.ndarray] = None,
        map_to_image_names: bool = True,
        **kwargs,
    ):
        coco_array = self._get_coco_array_from_path_or_array(coco_path, coco_array)

        perimeter_set_kwargs = {}
        for perimeter in self.perimeters:
            image_name_vs_referenced_perimeters = (
                perimeter.change_reference_with_coco_with_plural_references(
                    coco_array=coco_array, **kwargs
                )
            )
            for (
                image_name,
                referenced_perimeter,
            ) in image_name_vs_referenced_perimeters.items():
                if image_name in perimeter_set_kwargs:
                    perimeter_set_kwargs[image_name]["perimeters"].append(
                        referenced_perimeter
                    )
                else:
                    perimeter_set_kwargs[image_name] = {
                        "perimeters": [referenced_perimeter]
                    }

        if self.restricted_perimeters:
            for perimeter in self.restricted_perimeters:
                image_name_vs_referenced_perimeters = (
                    perimeter.change_reference_with_coco_with_plural_references(
                        coco_array, **kwargs
                    )
                )
                for (
                    image_name,
                    referenced_perimeter,
                ) in image_name_vs_referenced_perimeters.items():
                    if "restricted_perimeters" in perimeter_set_kwargs[image_name]:
                        perimeter_set_kwargs[image_name][
                            "restricted_perimeters"
                        ].append(referenced_perimeter)
                    else:
                        perimeter_set_kwargs[image_name] = {
                            "restricted_perimeters": [referenced_perimeter]
                        }

        if map_to_image_names:
            return {
                image_name: self.__class__(**perimeter_data)
                for image_name, perimeter_data in perimeter_set_kwargs.items()
            }
        return [
            self.__class__(**perimeter_data)
            for perimeter_data in perimeter_set_kwargs.values()
        ]

    def plot(self, **kwargs):
        ax = super().plot(**kwargs)
        return Perimeter.plot_perimeters(self.perimeters, ax)

    @cached_property
    def perimeter_vs_int_id(self):
        return {perimeter: perimeter.int_id for perimeter in self.perimeters}

    @cached_property
    def perimeter_vs_labels(self):
        return {perimeter: perimeter.labels for perimeter in self.perimeters}

    @cached_property
    def int_id_vs_label(self):
        return {perimeter.int_id: perimeter.labels for perimeter in self.perimeters}

    @property
    def _reference_point_variance(self):
        return statistics.variance(
            perimeter.reference_point for perimeter in self._all_perimeters
        )

    @property
    def _all_perimeters(self) -> Sequence:
        if not self.restricted_perimeters:
            return self.perimeters
        return *self.perimeters, *self.restricted_perimeters


def distance_between_two_perimeters(
    perimeter_a: Union[Perimeter, PerimeterSet],
    perimeter_b: Union[Perimeter, PerimeterSet],
):
    return np.linalg.norm(perimeter_a.centroid - perimeter_b.centroid)


def _coco_polygon_annotation(flat_annotation_data: Sequence):
    return [
        (flat_annotation_data[i], flat_annotation_data[i + 1])
        for i in range(0, len(flat_annotation_data) - 1, 2)
    ]


@lru_cache(50)
def _reference_point_from_coco_path(coco_path: Optional[FilePath]):
    coco_data = pd.read_csv(coco_path, header=None)
    assert len(coco_data) == 1
    return coco_data.values[0][1:3].astype(np.float64)
