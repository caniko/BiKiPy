from dataclasses import dataclass, field
from functools import cached_property, lru_cache
from logging import getLogger
from typing import Any, Sequence, SupportsFloat, SupportsInt, Union

import cv2
import matplotlib.pyplot as plt
import numpy as np

from bikipy.math.geometry import expand_parallelogram, order_parallelogram_corners
from bikipy.utils.video import get_video_data

logger = getLogger(__name__)


@dataclass
class Perimeter:
    """
    Parameters
    ----------
    int_label
        Integer-based label

    semantic_label: Optional, string
        String based semantic label for the perimeter. Useful during inspection and debuging

    inspect_image: Optional, string
        Label for the perimeter. Useful for manual audition and testing.
    """

    int_label: Union[int, None] = field(default=None)
    semantic_label: Union[str, None] = field(default=None)
    inspect_image: Union[str, None] = field(default=None, compare=False)

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
            ax.imshow(cv2.imread(str(self.inspect_image)))

        if points is not None:
            points = np.asarray(points)

            histogram, _x_edges, _y_edges = np.histogram2d(
                *points[np.logical_and(*np.isfinite(points).T)].T, bins=60
            )
            ax.imshow(histogram.T, interpolation="sinc")
            ax.plot(*points.T, ".r-")

        return ax


class PolygonalPerimeter(Perimeter):
    def __init__(
        self,
        perimeter_corners: Sequence[Sequence[SupportsFloat]],
        perimeter_border_normal_metric_magnitude: Union[SupportsFloat, None] = None,
        feature_scale: Union[Sequence[SupportsFloat], None] = None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.perimeter_corners = perimeter_corners
        self.feature_scale = feature_scale or None

        self.centroid = np.mean(perimeter_corners, axis=1)

        if perimeter_border_normal_metric_magnitude:
            self.perimeter_border_normal_metric_magnitude = (
                perimeter_border_normal_metric_magnitude
            )
        else:
            self._perimeter_border_normal_metric_magnitude = None
            self._border_corners = None

    def __getitem__(self, item: int):
        return self.perimeter_corners[item]

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
        return cls(
            perimeter_corners=perimeter_corners,
            inspect_image=inspect_image,
            *args,
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

    @property
    def perimeter_corners(self):
        return self._perimeter_corners

    @perimeter_corners.setter
    def perimeter_corners(self, corners: Sequence[Sequence[SupportsFloat]]):
        corners = np.asarray(corners)
        if (number_of_sides := corners.shape[0]) == 4:
            self._perimeter_corners = order_parallelogram_corners(corners)
        else:
            logger.warning(
                f"Number of corners, {number_of_sides}, not supported."
                f"The object methods may not work as intended."
            )
            self._perimeter_corners = corners

        self.number_of_sides = number_of_sides

    @cached_property
    def feat_scaled_sides(self):
        if not self.feature_scale:
            msg = "Feature scale parameters have not been defined in this instance"
            raise AttributeError(msg)
        return self.perimeter_corners / self.feature_scale

    @property
    def side_vectors(self):
        return self._corner_to_corner_vectors(self.perimeter_corners)

    @staticmethod
    def distance_between_two_vectors(border_a, border_b):
        return np.linalg.norm(border_a.centroid - border_b.centroid)

    @lru_cache
    def border(self, perimeter_border_normal_pixel_magnitude: Union[float, int]):
        """

        Parameters
        ----------
        perimeter_border_normal_pixel_magnitude
            The magnitude of the normal between the perimeter and the border given in pixels

        Returns
        -------

        """
        border_corners = expand_parallelogram(
            self.perimeter_corners, perimeter_border_normal_pixel_magnitude
        )
        border_vectors = self._corner_to_corner_vectors(border_corners)
        return border_corners, border_vectors

    def confined_coordinate_indexes(self, coordinates: Sequence):
        raise NotImplementedError

    def confined_coordinates(
        self, coordinates: Sequence, inspect: bool = False
    ) -> np.ndarray:
        """
        self.confined_coordinate_indexes to fetch confined coordinates within
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
            self.confined_coordinate_indexes(coordinates)
        ]
        if inspect:
            ax = super().plot()
            ax.scatter(confined_coordinates.T[0], confined_coordinates.T[1], marker="x")
            plt.show()

        return confined_coordinates

    @classmethod
    def detect_sequential_border_presence(
        cls,
        coordinates: Sequence[Sequence[SupportsFloat]],
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
            confined_coord_booleans_index = border.confined_coordinate_indexes(
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

        valid_indexes = np.nonzero(presence)
        if clean_outliers:
            presence = presence[valid_indexes]

        boolean_array = np.full(coordinates.shape[0], False, dtype=np.bool)
        boolean_array[valid_indexes] = True

        return presence, valid_indexes, boolean_array

    def plot(self, *args, include_borders: bool = False, **kwargs):
        """
        Plot the perimeter_corners defined in the object

        Returns
        -------
        matplotlib Axes object with the plot
        """
        ax = super().plot(*args, **kwargs)

        legends = []
        for i in range(len(self.perimeter_corners)):
            next = 0 if i + 1 == len(self.perimeter_corners) else i + 1

            side_a = self.perimeter_corners[i]
            side_b = self.perimeter_corners[next]
            ax.plot((side_a[0], side_b[0]), (side_a[1], side_b[1]), "o-")

            legend = [self._add_label_to_str(f"side {i}")]

            if include_borders:
                border_a = self.border_corners[i]
                border_b = self.border_corners[next]
                ax.plot((border_a[0], border_a[1]), (border_b[0], border_b[1]), "o-")

                legend.append(self._add_label_to_str(f"perimeter {i}"))

            legends.extend(legend)

        plt.legend(legends, bbox_to_anchor=(1.04, 0.5), loc="center left")
        return ax

    @staticmethod
    def _corner_to_corner_vectors(ordered_corners):
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


class GenericPolygonalBorder(PolygonalPerimeter):
    @property
    def sides(self):
        return self.__sides
