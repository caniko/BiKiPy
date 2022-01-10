import copy
import statistics
from dataclasses import InitVar, dataclass
from functools import cached_property
from logging import getLogger
from pathlib import Path
from typing import Any, ClassVar, Optional, Sequence, Union

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray as NpNDArray
from pydantic import DirectoryPath, Field, FilePath, root_validator, validator
from shapely.geometry import Point, Polygon

from bikipy.core.base_class import BikipyBaseHashable
from bikipy.math.geometry import clockwise_sort_points, expand_bikipy_perimeter
from bikipy.math.vector import normal_from_line_to_point, point_to_line_segment_distance
from bikipy.utils.misc import (
    get_reference_point_from_array,
    read_image,
    read_makesense_point_csv,
)
from bikipy.utils.typing import NDArray

logger = getLogger(__name__)


class BasePerimeter(BikipyBaseHashable):
    reference_point_coco_path: Optional[FilePath] = None
    reference_point_array: Optional[NDArray] = None
    inspect_image_path: Optional[FilePath] = None
    inspect_image_array: Optional[NDArray] = None

    category: ClassVar[Optional[str]] = "perimeter"

    @root_validator(pre=True)
    def mutually_exclusive(cls, values):
        if all(key in values for key in ("inspect_image_path", "inspect_image_array")):
            msg = "inspect_image_path and inspect_image_array must be defined mutually exclusive"
            raise AttributeError(msg)
        if all(
            key in values
            for key in ("reference_point_coco_path", "reference_point_array")
        ):
            msg = "reference_point_coco_path and reference_point_array must be defined mutually exclusive"
            raise AttributeError(msg)
        return values

    @property
    def inspect_image(self):
        if self.inspect_image_array is None and not self.inspect_image_path:
            return None
        return (
            read_image(self.inspect_image_path)
            if self.inspect_image_path
            else self.inspect_image_array
        )

    @inspect_image.setter
    def inspect_image(self, value):
        self.inspect_image_array = np.asarray(value)

    @property
    def reference_point(self):
        from bikipy.perimeter.io.makesense import reference_point_from_coco_path

        if self.reference_point_array is None and not self.reference_point_coco_path:
            return None
        return (
            reference_point_from_coco_path(self.reference_point_coco_path)
            if self.reference_point_array is None
            else self.reference_point_array
        )

    @reference_point.setter
    def reference_point(self, value):
        self.reference_point_array = np.asarray(value)

    def plot(self, ax: Any = None, coordinates: Optional[Sequence] = None):
        """
        Plot the perimeter using matplotlib. Optionally, plot coordinates alongside the perimeter

        Parameters
        ----------
        ax
            Axes object that the plot will be saved in. A new instance of Axes will be used
            if object returns False.
        coordinates
            Sequence of 2D coordinates that will be plotted alongside the perimeter

        Returns
        -------
        Axes object with plots
        """
        if not ax:
            fig, ax = plt.subplots()

        if self.inspect_image is not None:
            ax.imshow(self.inspect_image)

        if coordinates is not None:
            coordinates = np.asarray(coordinates)

            histogram, _x_edges, _y_edges = np.histogram2d(
                *coordinates[np.logical_and(*np.isfinite(coordinates).T)].T, bins=60
            )
            ax.imshow(histogram.T, interpolation="sinc")
            ax.plot(*coordinates.T, ".r-")

        ax.set_title(self.best_id)

        return ax

    @staticmethod
    def _get_coco_array_from_path_or_array(**kwargs):
        return _get_coco_array_from_path_or_array(**kwargs)


class Perimeter(BasePerimeter):
    corners: np.ndarray
    reference_point_coco_path: Optional[FilePath] = None
    reference_point_array: Optional[NDArray] = None
    inspect_image_path: Optional[FilePath] = None
    inspect_image_array: Optional[NDArray] = None
    feature_scale: Optional[NDArray] = None

    category: ClassVar[Optional[str]] = "perimeter"

    _polygon_order: ClassVar[Optional[int]] = None

    @root_validator(pre=True)
    def mutually_exclusive(cls, values):
        if all(key in values for key in ("inspect_image_path", "inspect_image_array")):
            msg = "inspect_image_path and inspect_image_array must be defined mutually exclusive"
            raise AttributeError(msg)
        if all(
            key in values
            for key in ("reference_point_coco_path", "reference_point_array")
        ):
            msg = "reference_point_coco_path and reference_point_array must be defined mutually exclusive"
            raise AttributeError(msg)
        return values

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

    def expand(self, perimeter_border_normal_pixel_magnitude: Union[float, int]):
        """
        :param perimeter_border_normal_pixel_magnitude: The magnitude of the normal between
            the perimeter and the perimeter given in pixels
        :return:
        """
        if self._polygon_order == 4:
            from bikipy.perimeter import ParallelogramPerimeter

            border_obj = ParallelogramPerimeter(
                corners=expand_bikipy_perimeter(
                    self, perimeter_border_normal_pixel_magnitude
                ),
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

        closest_corner_start_point = np.zeros(
            (closest_distance.shape[0], 2), dtype=np.float32
        )
        closest_corner_vectors = np.zeros(
            (closest_distance.shape[0], 2), dtype=np.float32
        )
        for i in range(self.number_of_corners):
            closest_corner_start_point[closest_index == i] = self.corners[i]
            closest_corner_vectors[
                closest_index == i
            ] = self.perimeter_corner_to_next_clockwise_corner_vectors[i]

        return closest_corner_start_point, closest_corner_vectors

    def closest_perimeter_points_to_coordinates(self, coordinates: Sequence):
        (
            closest_corner_start_point,
            closest_corner_vectors,
        ) = self.closest_sides_to_coordinates(coordinates)

        return normal_from_line_to_point(
            closest_corner_vectors, closest_corner_start_point, coordinates
        )

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

        perimeter_sequence = (
            (*inferior_poly_border_instances, *superior_poly_border_instances)
            if inferior_poly_border_instances
            else superior_poly_border_instances
        )
        presence = np.zeros(
            coordinates.shape[0],
            dtype=np.int8 if len(perimeter_sequence) <= 255 else np.int16,
        )

        overlap_locations = {}
        for perimeter in perimeter_sequence:
            confined_coord_booleans_index = (
                perimeter.coordinate_confinement_boolean_index(coordinates)
            )

            if presence[confined_coord_booleans_index].any():
                overlap_locations[perimeter.label] = np.flatnonzero(
                    presence[confined_coord_booleans_index]
                )
                presence[overlap_locations[perimeter.label]] = 0
                logger.info(
                    f"BasePerimeter {perimeter.label} has coordinate overlap with "
                    f"other border_corners, {overlap_locations[perimeter.label].size}"
                )

            presence[confined_coord_booleans_index] = perimeter.int_id

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
        metadata_path: Optional[FilePath] = None,
        coco_array: Optional[np.ndarray] = None,
        **kwargs,
    ):
        coco_array = _get_coco_array_from_path_or_array(metadata_path, coco_array)

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
        metadata_path: Optional[FilePath] = None,
        coco_array: Optional[np.ndarray] = None,
        image_root: Optional[DirectoryPath] = None,
        map_to_image_names: bool = True,
    ):
        def _change_reference_loop_func(reference_point, img_name):
            return self.change_reference(
                reference_point,
                new_inspect_image_path=image_root / img_name if image_root else None,
            )

        coco_array = _get_coco_array_from_path_or_array(metadata_path, coco_array)

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
                perimeter = self.perimeter(perimeter_border_normal_pixel_magnitude)
                border_a = perimeter[index]
                border_b = perimeter[following_index]
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
    def perimeter_corner_to_next_clockwise_corner_vectors(self):
        return np.diff(self.corners[::-1], prepend=[self.corners[0]], axis=0)[::-1]

    @cached_property
    def perimeter_lengths(self):
        return np.linalg.norm(
            self.perimeter_corner_to_next_clockwise_corner_vectors, axis=1
        )

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

    def _add_label_to_str(self, in_string):
        if self.label:
            return f"{self.label} {in_string}"
        if self.int_id:
            return f"{self.int_id} {in_string}"
        return in_string


@dataclass
class PerimeterSet:
    perimeters: tuple
    restricted_perimeters: Optional[tuple] = None

    category: ClassVar[Optional[str]] = "perimeter"

    class Config:
        arbitrary_types_allowed = True
        keep_untouched = (cached_property,)

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

    def change_reference(self, **perimeter_change_reference_kwargs):
        return self.__class__(
            perimeters=tuple(
                perimeter.change_reference(**perimeter_change_reference_kwargs)
                for perimeter in self.perimeters
            ),
            restricted_perimeters=tuple(
                perimeter.change_reference(**perimeter_change_reference_kwargs)
                for perimeter in self.restricted_perimeters
            ),
        )

    def change_reference_with_coco(
        self,
        metadata_path: Optional[FilePath] = None,
        coco_array: Optional[np.ndarray] = None,
    ):
        coco_array = _get_coco_array_from_path_or_array(metadata_path, coco_array)

        if len(coco_array) != 1:
            msg = (
                "The coco array includes more than one annotation. "
                "Please use change_reference_with_coco_with_plural_references()"
            )
            raise ValueError(msg)

        return self.__class__(
            perimeters=self.perimeters,
            restricted_perimeters=self.restricted_perimeters,
            reference_point_array=coco_array,
        )

    def change_reference_with_coco_with_plural_references(
        self,
        metadata_path: Optional[FilePath] = None,
        coco_array: Optional[np.ndarray] = None,
        map_to_image_names: bool = True,
        **kwargs,
    ):
        coco_array = _get_coco_array_from_path_or_array(metadata_path, coco_array)

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

        for perimeter in self.restricted_perimeters or []:
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
                    perimeter_set_kwargs[image_name]["restricted_perimeters"].append(
                        referenced_perimeter
                    )
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
    def reference_point(self):
        expected_reference_point = self._all_perimeters[0].reference_point
        if equality := np.all(
            expected_reference_point == perimeter.reference_point
            for perimeter in self._all_perimeters
        ):
            logger.warning(
                "The reference points are different within the perimeter set"
            )
        if not equality or not np.any(expected_reference_point):
            return None
        return expected_reference_point

    @property
    def inspect_image(self):
        result = self.perimeters[0].inspect_image
        assert all(
            result == perimeter.inspect_image for perimeter in self._all_perimeters
        )
        return result

    @property
    def inspect_image_path(self):
        result = self.perimeters[0].inspect_image_path
        assert all(
            result == perimeter.inspect_image_path for perimeter in self._all_perimeters
        )
        return result

    @property
    def _all_perimeters(self) -> Sequence:
        if not self.restricted_perimeters:
            return self.perimeters
        return (
            *self.perimeters,
            *self.restricted_perimeters,
        )


def distance_between_two_perimeters(
    perimeter_a: Union[Perimeter, PerimeterSet],
    perimeter_b: Union[Perimeter, PerimeterSet],
):
    return np.linalg.norm(perimeter_a.centroid - perimeter_b.centroid)


def _get_coco_array_from_path_or_array(
    metadata_path: Optional[FilePath] = None,
    coco_array: Optional[np.ndarray] = None,
):
    msg = "metadata_path and coco_array are defined mutually exclusive"
    if metadata_path and np.any(coco_array):
        raise ValueError(msg)

    if metadata_path:
        result = read_makesense_point_csv(metadata_path)
    elif np.any(coco_array):
        result = coco_array
    else:
        raise ValueError(msg)

    assert np.any(result)
    return result
