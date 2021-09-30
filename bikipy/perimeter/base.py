import copy
import json
import statistics
from functools import cached_property, lru_cache
from logging import getLogger
from pathlib import PurePath, Path
from typing import Any, Sequence, Union

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numba import jit
from shapely.geometry import Point, Polygon

from bikipy.math.geometry import expand_parallelogram
from bikipy.math.vector import point_to_line_segment_distance
from bikipy.utils.misc import read_image
from bikipy.utils.video import get_video_data

logger = getLogger(__name__)


class Perimeter:
    def __init__(
        self,
        int_label: Union[int, None] = None,
        semantic_label: Union[str, None] = None,
        reference_point: Union[np.ndarray, Sequence[float], None] = None,
        inspect_image: Union[str, PurePath, np.ndarray, None] = None,
    ):
        """
        :param int_label: Integer label
        :param semantic_label: String/semantic label. Useful during inspection and debuging
        :param inspect_image: Label for the perimeter. Useful for manual audition and testing.
        :type int_label: int
        :type semantic_label: str
        :type inspect_image: Any
        """

        self.int_label = int(int_label) if int_label else None
        self.semantic_label = str(semantic_label) if semantic_label else None

        self.inspect_image = inspect_image
        self._reference_point = None
        self._inspect_image_path = None
        self.reference_point = reference_point

    @property
    def inspect_image(self):
        return self._inspect_image

    @inspect_image.setter
    def inspect_image(self, value):
        if not isinstance(value, np.ndarray):
            if (value := Path(value)).exists():
                self.inspect_image_path = value
                return
            msg = (
                "The provided object is not a numpy array; it is not an image."
                "In case it is a path, it does not exist"
            )
            raise ValueError(msg)

        self._inspect_image = value

    @property
    def inspect_image_path(self):
        return self._inspect_image_path

    @inspect_image_path.setter
    def inspect_image_path(self, value: Union[PurePath, str]):
        if not (value := Path(value)).exists():
            msg = (
                f"Failed to read inspect_image, the provided path does not exist.\n"
                f"Path: {value}"
            )
            raise ValueError(msg)
        self.inspect_image = read_image(value, 0)
        self._inspect_image_path = value

    def plot(self, ax: Any = None, points: Union[Sequence, None] = None, **kwargs):
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
            ax.imshow(read_image(self.inspect_image), cmap="gray", vmin=0, vmax=255)

        if points is not None:
            points = np.asarray(points)

            histogram, _x_edges, _y_edges = np.histogram2d(
                *points[np.logical_and(*np.isfinite(points).T)].T, bins=60
            )
            ax.imshow(histogram.T, interpolation="sinc")
            ax.plot(*points.T, ".r-")

        return ax

    @property
    def reference_point(self):
        return self._reference_point

    @reference_point.setter
    def reference_point(self, value):
        value = np.asarray(value, dtype=np.float32)
        if value is not None and self._reference_point is not None:
            if np.all(self._reference_point == value):
                logger.info("The provided reference_point is identical to the current")
            else:
                self.change_reference_function(value)
        self._reference_point = value

    def change_reference_function(self, new_reference: np.ndarray):
        msg = (
            f"There was a change in reference_point. However, the change_reference_function function "
            f"has not been implemented in {self.__class__}"
        )
        logger.warning(msg)
        raise NotImplemented(msg)

    def change_reference_with_image(self, image: Union[PurePath, str, np.ndarray]):
        self.reference_point = self._annotate_reference(image)

    @staticmethod
    def _annotate_reference(image: Union[PurePath, str, np.ndarray]):
        plt.imshow(image if isinstance(image, np.ndarray) else cv2.imread(str(image)))
        plt.title("Please click on the reference_point point")
        return np.array(plt.ginput(n=1, timeout=0)[0])

    def change_reference_with_coco(self, coco_path: Union[PurePath, str], image_root: Union[PurePath, str, None] = None):
        def get_reference_point_from_array(array: np.ndarray):
            return array[1:3]

        if not self.reference_point:
            msg = "The current object has no defined reference_point"
            raise AttributeError(msg)

        csv_array = pd.read_csv(
            coco_path,
            header=None,
            # names=["x1", "y1", "x2", "y2", "filename", "img_x", "img_y"],
        ).to_numpy()

        if (number_of_references := len(csv_array)) == 1:
            return get_reference_point_from_array(csv_array)
        elif number_of_references > 1:
            result = []
            for row in csv_array:
                new = copy.deepcopy(self)
                new.reference_point = get_reference_point_from_array(row)
                new.inspect_image
                result.append(new)
        else:
            msg = f"The path in coco_path yields an empty dataset"
            raise ValueError(msg)

    def __repr__(self):
        return (
            f"{self.__class__.__name__}\n\t"
            f"ID={self.int_label}; label={self.semantic_label}\n\t"
            f"inspect_image={self.inspect_image is not None}\n\t"
            f"label={self.semantic_label}"
        )


class PolygonalPerimeter(Perimeter):
    def __init__(
        self,
        corners: Sequence[Sequence[float]],
        feature_scale: Union[Sequence[float], None] = None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.corners = np.asarray(corners)
        self.number_of_sides = self.corners.shape[0]
        self.feature_scale = feature_scale or None

    def __getitem__(self, item: int):
        return self.corners[item]

    def __repr__(self):
        return super().__repr__() + f"\n\tcorners={self.corners}"

    @classmethod
    def init_polygon(cls, corners: Sequence, **kwargs):
        corners = np.asarray(corners)
        if (number_of_sides := corners.shape[0]) == 3:
            from bikipy.perimeter.triangular import TriangularPerimeter

            return TriangularPerimeter(corners=corners, **kwargs)
        elif number_of_sides == 4:
            from bikipy.perimeter.parallelogram.classes import ParallelogramPerimeter

            return ParallelogramPerimeter(corners=corners, **kwargs)
        else:
            return cls(corners=corners, **kwargs)

    @classmethod
    def from_image(cls, inspect_image: Any, n: int, *args, **kwargs):
        """
        Define the corners of a polygon with a guiding image

        Parameters
        ----------
        inspect_image
            Either path to image or image in numpy array
        n
            The number of sides on the polygon. Each side has to be annotated

        Returns
        -------
        list with pixel coordinates of the polygon corners
        """
        logger.debug("Generating PolygonalPerimeter from image data")

        plt.imshow(
            inspect_image
            if isinstance(inspect_image, np.ndarray)
            else cv2.imread(str(inspect_image))
        )

        return cls.init_polygon(
            corners=plt.ginput(n=n, timeout=0),
            inspect_image=inspect_image,
            **kwargs,
        )

    @classmethod
    def from_video(cls, video_path: Any, frame_time: str = "middle", *args, **kwargs):
        """
        Initialize class using a frame from a sample video file

        Parameters
        ----------
        video_path: str
            The path to the video file

        frame_time: str
            Relative location of the frame used for reference_point in analysis

        Returns
        -------
        bikipy.perimeter.polygon.polygon_corners_on_image call with frame from video
        """
        logger.debug("Generating PolygonalPerimeter from video data")

        frame, x_res, y_res, _fps = get_video_data(video_path, frame_time)

        return cls.from_image(frame, *args, feature_scale=(x_res, y_res), **kwargs)

    @classmethod
    def from_coco(
        cls,
        coco_path: Any,
        reference_point_annotation: bool = False,
        single_obj_return: bool = False,
        **kwargs,
    ) -> Union[dict, Perimeter]:
        logger.debug("Generating PolygonalPerimeter from coco data")

        with open(coco_path, "rb") as in_json:
            coco = json.load(in_json)

        # The coco annotations are not sorted with respect to the category IDs
        coco["annotations"] = sorted(
            coco["annotations"], key=lambda dictionary: dictionary["category_id"]
        )

        # We don't need to do this, but better to be on the safe side
        coco["categories"] = sorted(
            coco["categories"], key=lambda dictionary: dictionary["id"]
        )

        if reference_point_annotation:
            if "inspect_image" not in kwargs:
                msg = "inspect_image is not defined"
                raise ValueError(msg)
            reference_point = cls._annotate_reference(kwargs["inspect_image"])
        else:
            reference_point = False

        results = {}
        for annotation, category in zip(coco["annotations"], coco["categories"]):
            assert int(annotation["category_id"]) == int(
                category["id"]
            ), f"{annotation['category_id']} != {category['id']}"

            segmentation = annotation["segmentation"][0]
            results[category["name"].lower()] = cls.init_polygon(
                [  # corners
                    (segmentation[i], segmentation[i + 1])
                    for i in range(0, len(segmentation) - 1, 2)
                ],
                reference_point=reference_point,
                **kwargs,
            )

        if single_obj_return:
            assert len(results) == 1, f"More than one item in coco set, {len(results)}"
            return results.popitem()[1]
        return results

    @cached_property
    def perimeter_vectors(self):
        return np.diff(self.corners[::-1], prepend=[self.corners[0]], axis=0)[::-1]

    @cached_property
    def line_segment_pairs(self):
        pairs = [
            (self.corners[i], self.corners[i + 1])
            for i in range(self.number_of_sides - 1)
        ]
        pairs.append((self.corners[-1], self.corners[0]))
        return np.array(pairs)

    @cached_property
    def centroid(self):
        return np.mean(self.corners, axis=1)

    @lru_cache
    def border(
        self,
        perimeter_border_normal_pixel_magnitude: Union[float, int],
    ):
        """
        :param perimeter_border_normal_pixel_magnitude: The magnitude of the normal between
            the perimeter and the border given in pixels
        :return:
        """
        border_obj = self.__class__(
            expand_parallelogram(self.corners, perimeter_border_normal_pixel_magnitude),
            inspect_image=self.inspect_image,
        )

        return border_obj

    def confined_coordinates(self, coordinates: Sequence, inspect: bool = False):
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
        if inspect:
            ax = self.plot_self()
            ax.scatter(
                coordinate_confinement_boolean_index.T[0],
                coordinate_confinement_boolean_index.T[1],
                marker="x",
            )
            ax.set_tittle("Confined coordinates")
            plt.show()

        return coordinate_confinement_boolean_index

    @lru_cache
    @jit
    def coordinate_confinement_boolean_index(self, coordinates: Sequence) -> np.ndarray:
        assert self.number_of_sides > 4

        polygon = Polygon(self.corners)
        return np.array(
            [polygon.contains(Point(coordinate)) for coordinate in coordinates]
        )

    @classmethod
    def detect_sequential_border_presence(
        cls,
        coordinates: Sequence[Sequence[float]],
        superior_poly_border_instances: Union[Sequence, None],
        inferior_poly_border_instances: Union[Sequence, None] = None,
        clean_outliers: bool = True,
    ):
        """
        Define sequential perimeter confinements of coordinates

        Parameters
        ----------
        coordinates: Sequence
            Coordinates that will have their confinement tested

        superior_poly_border_instances: Sequence
            PolygonalPerimeter instances that will have the highest priority
            in case of overlap with respect to confinement

        inferior_poly_border_instances: Sequence
            PolygonalPerimeter instances that will have the lowest priority
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
            dtype=np.int8 if len(border_sequence) <= 7 else np.int16,
        )
        overlap_locations = {}

        for border in border_sequence:
            confined_coord_booleans_index = border.coordinate_confinement_boolean_index(
                coordinates
            )

            if presence[confined_coord_booleans_index].any():
                overlap_locations[border.semantic_label] = np.flatnonzero(
                    presence[confined_coord_booleans_index]
                )
                presence[overlap_locations[border.semantic_label]] = 0
                logger.info(
                    f"Perimeter {border.semantic_label} has coordinate overlap with "
                    f"other border_corners, {overlap_locations[border.semantic_label].size}"
                )

            presence[confined_coord_booleans_index] = border.int_label

        valid_indices = np.nonzero(presence)
        if clean_outliers:
            presence = presence[valid_indices]

        boolean_array = np.full(coordinates.shape[0], False, dtype=np.bool)
        boolean_array[valid_indices] = True

        return presence, valid_indices, boolean_array

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
        for i in range(self.number_of_sides):
            closest_vectors[closest_index == i] = self.perimeter_vectors[i]

        return closest_distance, closest_vectors

    def change_reference_function(self, new_reference: np.ndarray):
        self.corners += new_reference - self.reference_point

    @staticmethod
    def distance_between_two_perimeters(perimeter_a, perimeter_b):
        return np.linalg.norm(perimeter_a.centroid - perimeter_b.centroid)

    def plot_self(
        self,
        plot_kwargs: Union[dict, None] = None,
        perimeter_plot_kwargs: Union[dict, None] = None,
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
        perimeter_plot_kwargs: Union[dict, None] = None,
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
                label=self.semantic_label,
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
    def linked_corners(self):
        return np.append(self.corners, np.expand_dims(self.corners[0], 0), axis=0)

    @cached_property
    def edge_midpoints(self):
        return self.corners + np.diff(self.linked_corners, axis=0) / 2.0

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

    def _add_label_to_str(self, in_string):
        if self.semantic_label:
            return f"{self.semantic_label} {in_string}"
        if self.int_label:
            return f"{self.int_label} {in_string}"
        return in_string


class GenericPolygonalBorder(PolygonalPerimeter):
    # Deprecated.
    @property
    def sides(self):
        return self.__sides


PERIMETER_SEQUENCE_OR_DICT = Union[
    Sequence[PolygonalPerimeter], dict[str, Sequence[PolygonalPerimeter]]
]


class PolygonalPerimeterSet(Perimeter):
    def __init__(
        self,
        perimeters: PERIMETER_SEQUENCE_OR_DICT,
        restricted_perimeters: Union[PERIMETER_SEQUENCE_OR_DICT, None] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        assert all(
            isinstance(perimeter, PolygonalPerimeter) for perimeter in perimeters
        )

        self.perimeters = perimeters
        self.restricted_perimeters = restricted_perimeters

    def combined_contained_coordinates(self, coordinates: Sequence):
        present = np.any(
            [
                perimeter.coordinate_confinement_boolean_index(coordinates)
                for perimeter in self._perimeter_iterable
            ]
        )
        if self.restricted_perimeters:
            present = present & ~np.any(
                [
                    perimeter.coordinate_confinement_boolean_index(coordinates)
                    for perimeter in self._restricted_perimeters_iterable
                ]
            )
        return present

    def change_reference_function(self, new_reference: np.ndarray):
        for perimeter in self.perimeters:
            perimeter.reference_point = new_reference
        for perimeter in self.restricted_perimeters:
            perimeter.reference_point = new_reference

    @property
    def _reference_point_variance(self):
        return statistics.variance(
            perimeter.reference_point for perimeter in self._all_perimeters
        )

    def plot(self, **kwargs):
        ax = super().plot(**kwargs)
        return PolygonalPerimeter.plot_perimeters(self.perimeters, ax)

    @cached_property
    def _all_perimeters(self):
        return *self._perimeter_iterable, *self._restricted_perimeters_iterable

    @cached_property
    def _perimeter_is_dict(self):
        return isinstance(self.perimeters, dict)

    @cached_property
    def _perimeter_iterable(self):
        return self.perimeters.values() if self._perimeter_is_dict else self.perimeters

    @cached_property
    def _restricted_perimeters_is_dict(self):
        return isinstance(self.restricted_perimeters, dict)

    @cached_property
    def _restricted_perimeters_iterable(self):
        return (
            self.restricted_perimeters.values()
            if self._restricted_perimeters_is_dict
            else self.restricted_perimeters
        )

    @cached_property
    def _unified_dict(self):
        if self._perimeter_is_dict and self._restricted_perimeters_is_dict:
            return {**self.perimeters, **self.restricted_perimeters}
        if self._perimeter_is_dict:
            return self.perimeters
        if self._restricted_perimeters_is_dict:
            return self.restricted_perimeters
        msg = "None of the perimeter datastructures are mappable"
        raise AttributeError(msg)

    @lru_cache
    def __getitem__(self, item: Union[str, int]):
        try:
            return self._unified_dict[item]
        except AttributeError:
            for perimeter in self._all_perimeters:
                if perimeter.semantic_label == item or perimeter.int_label == item:
                    return perimeter
        msg = f"Item was not found, {item}"
        raise KeyError(msg)


Perimeter2D = Union[PolygonalPerimeter, PolygonalPerimeterSet]
