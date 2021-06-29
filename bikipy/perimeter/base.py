import json
from functools import cached_property, lru_cache
from logging import getLogger
from pathlib import PurePath
from typing import Any, Sequence, Union

import cv2
import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import Point, Polygon

from bikipy.math.geometry import expand_parallelogram, order_parallelogram_corners
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

        if inspect_image is not None:
            self.inspect_image = inspect_image
        else:
            self._inspect_image = None

    @property
    def inspect_image(self):
        return self._inspect_image

    @inspect_image.setter
    def inspect_image(self, value):
        self._inspect_image = read_image(value, 0)

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

        self.perimeter_corners = perimeter_corners
        self.feature_scale = feature_scale or None

        self.centroid = np.mean(perimeter_corners, axis=1)

    def __getitem__(self, item: int):
        return self.perimeter_corners[item]

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
        with json.load(coco_path) as coco:
            annotations = coco["annotations"]

        results = {}
        for annotation in annotations:
            segmentation = annotation["segmentation"]
            results[annotation["name"].lower()] = cls.init_polygon(
                [  # perimeter_corners
                    (segmentation[i], segmentation[i + 1])
                    for i in range(0, len(segmentation) - 1, 2)
                ],
                *kwargs,
            )

        return results

    @property
    def perimeter_corners(self):
        return self._perimeter_corners

    @perimeter_corners.setter
    def perimeter_corners(self, corners: Sequence[Sequence[float]]):
        corners = np.asarray(corners)

        self._perimeter_corners = order_parallelogram_corners(corners)
        self.number_of_sides = corners.shape[0]

    @cached_property
    def feat_scaled_sides(self):
        if not self.feature_scale:
            msg = "Feature scale parameters have not been defined in this instance"
            raise AttributeError(msg)
        return self.perimeter_corners / self.feature_scale

    @property
    def corner_to_corner_vectors(self):
        return self._vectors_from_neighboring_points(self.perimeter_corners)

    @staticmethod
    def distance_between_two_vectors(border_a, border_b):
        return np.linalg.norm(border_a.centroid - border_b.centroid)

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

    def confined_coordinate_indices(self, coordinates: Sequence):
        assert self.number_of_sides > 4

        polygon = Polygon(self.perimeter_corners)
        return np.array(
            [polygon.contains(Point(coordinate)) for coordinate in coordinates]
        )

    def confined_coordinates(
        self, coordinates: Sequence, inspect: bool = False
    ) -> np.ndarray:
        """
        self.confined_coordinate_indices to fetch confined coordinates within
        the respective perimeter

        Parameters
        ----------
        coordinates: Sequence
            Coordinates that will have their confinement tested

        inspect: bool
            If True, plot the confined coordinates

        Returns
        -------
        np.ndarray, confined coordinates
        """

        coordinates = np.asarray(coordinates)
        confined_coordinates = coordinates[
            self.confined_coordinate_indices(coordinates)
        ]
        if inspect:
            ax = super().plot()
            ax.scatter(confined_coordinates.T[0], confined_coordinates.T[1], marker="x")
            ax.set_tittle("Confined coordinates")
            plt.show()

        return confined_coordinates

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
            confined_coord_booleans_index = border.confined_coordinate_indices(
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

    @staticmethod
    def _vectors_from_neighboring_points(ordered_corners):
        return (
            *np.diff(ordered_corners, axis=0),
            ordered_corners[0] - ordered_corners[-1],
        )

    def _add_label_to_str(self, in_string):
        if self.semantic_label:
            return f"{self.semantic_label} {in_string}"
        if self.int_label:
            return f"{self.int_label} {in_string}"

        return in_string

    def __repr__(self):
        return super().__repr__() + f"\n\tperimeter_corners={self.perimeter_corners}"


class GenericPolygonalBorder(PolygonalPerimeter):
    # Deprecated.
    @property
    def sides(self):
        return self.__sides
