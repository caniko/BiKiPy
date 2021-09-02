import json
from functools import cached_property, lru_cache
from logging import getLogger
from pathlib import PurePath
from typing import Any, Sequence, Union

import cv2
import matplotlib.pyplot as plt
import numpy as np
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

        self._inspect_image = None

    @property
    def inspect_image(self):
        return self._inspect_image

    @inspect_image.setter
    def inspect_image(self, value):
        self._inspect_image = read_image(value, 0) if value else None

    def plot(self, ax: Any = None, points: Union[Sequence, None] = None):
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
        perimeter_corners: Sequence[Sequence[float]],
        feature_scale: Union[Sequence[float], None] = None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.perimeter_corners = np.asarray(perimeter_corners)

        self.number_of_sides = self.perimeter_corners.shape[0]
        self.centroid = np.mean(perimeter_corners, axis=1)

        self.feature_scale = feature_scale or None

    def __getitem__(self, item: int):
        return self.perimeter_corners[item]

    def __repr__(self):
        return super().__repr__() + f"\n\tperimeter_corners={self.perimeter_corners}"

    @classmethod
    def init_polygon(cls, perimeter_corners: Sequence, **kwargs):
        perimeter_corners = np.asarray(perimeter_corners)
        if (number_of_sides := perimeter_corners.shape[0]) == 3:
            from bikipy.perimeter.triangular import TriangularPerimeter

            return TriangularPerimeter(perimeter_corners=perimeter_corners, **kwargs)
        elif number_of_sides == 4:
            from bikipy.perimeter.parallelogram.classes import ParallelogramPerimeter

            return ParallelogramPerimeter(perimeter_corners=perimeter_corners, **kwargs)
        else:
            return cls(perimeter_corners=perimeter_corners, **kwargs)

    @classmethod
    def from_image(cls, inspect_image: Any, n: int, *args, **kwargs):
        """
        Define the corners of a polygon with a guiding image

        Parameters
        ----------
        inspect_image
            Either path to image or PIL.Image object with opened image inside
        n
            The number of sides on the polygon. Each side has to be annotated

        Returns
        -------
        list with pixel coordinates of the polygon corners
        """

        if isinstance(inspect_image, np.ndarray):
            img = inspect_image
        else:
            img = cv2.imread(str(inspect_image))
        plt.imshow(img)

        perimeter_corners = plt.ginput(n=n, timeout=0)
        return cls.init_polygon(
            perimeter_corners=perimeter_corners,
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
            Relative location of the frame used for reference in analysis

        Returns
        -------
        bikipy.perimeter.polygon.polygon_corners_on_image call with frame from video
        """

        frame, x_res, y_res, _fps = get_video_data(video_path, frame_time)

        return cls.from_image(frame, *args, feature_scale=(x_res, y_res), **kwargs)

    @classmethod
    def from_coco(cls, coco_path: Any, **kwargs) -> dict:
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

        results = {}
        for annotation, category in zip(coco["annotations"], coco["categories"]):
            assert int(annotation["category_id"]) == int(category["id"]), f"{annotation['category_id']} != {category['id']}"
            segmentation = annotation["segmentation"][0]
            results[category["name"].lower()] = cls.init_polygon(
                [  # perimeter_corners
                    (segmentation[i], segmentation[i + 1])
                    for i in range(0, len(segmentation) - 1, 2)
                ],
                **kwargs,
            )

        return results

    @cached_property
    def perimeter_vectors(self):
        return np.diff(
            self.perimeter_corners[::-1], prepend=[self.perimeter_corners[0]], axis=0
        )[::-1]

    @cached_property
    def line_segment_pairs(self):
        pairs = [
            (self.perimeter_corners[i], self.perimeter_corners[i + 1])
            for i in range(self.number_of_sides - 1)
        ]
        pairs.append((self.perimeter_corners[-1], self.perimeter_corners[0]))
        return np.array(pairs)

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
            expand_parallelogram(
                self.perimeter_corners, perimeter_border_normal_pixel_magnitude
            ),
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
            ax = super().plot()
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

        polygon = Polygon(self.perimeter_corners)
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
        closest_vectors = np.zeros((closest_distance.shape[0], 2), dtype=np.float)
        for i in range(self.number_of_sides):
            closest_vectors[closest_index == i] = self.perimeter_vectors[i]

        return closest_distance, closest_vectors

    @staticmethod
    def distance_between_two_vectors(border_a, border_b):
        return np.linalg.norm(border_a.centroid - border_b.centroid)

    def plot(
        self,
        perimeter_border_normal_pixel_magnitude: Union[float, int, None] = None,
        **kwargs,
    ):
        """
        Plot the perimeter_corners defined in the object

        Returns
        -------
        matplotlib Axes object with the plot
        """
        ax = super().plot(**kwargs)

        legends = []
        for index in range(len(self.perimeter_corners)):
            following_index = (
                0 if index + 1 == len(self.perimeter_corners) else index + 1
            )

            corner_a = self.perimeter_corners[index]
            corner_b = self.perimeter_corners[following_index]
            ax.plot((corner_a[0], corner_b[0]), (corner_a[1], corner_b[1]), "o-")

            legend = [self._add_label_to_str(f"side {index}")]

            if perimeter_border_normal_pixel_magnitude:
                border = self.border(perimeter_border_normal_pixel_magnitude)
                border_a = border[index]
                border_b = border[following_index]
                ax.plot((border_a[0], border_b[0]), (border_a[1], border_b[1]), "o-")

                legend.append(self._add_label_to_str(f"perimeter {index}"))

            legends.extend(legend)

        # plt.legend(legends, bbox_to_anchor=(1.04, 0.5), loc="center left")
        return ax

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


class CombinedPolygonalPerimeter(Perimeter):
    def __init__(
        self,
        rectangles: Sequence[PolygonalPerimeter],
        restrict_zones: Union[Sequence[PolygonalPerimeter], None] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.rectangles = rectangles
        self.restrict_zones = restrict_zones

    def contained_coordinates(self, coordinates: Sequence):
        present = np.any(
            [
                rectangle.coordinate_confinement_boolean_index(coordinates)
                for rectangle in self.rectangles
            ]
        )
        if self.restrict_zones:
            present = present & ~np.any(
                [
                    rectangle.coordinate_confinement_boolean_index(coordinates)
                    for rectangle in self.restrict_zones
                ]
            )
        return present


Perimeter2D = Union[PolygonalPerimeter, CombinedPolygonalPerimeter]
