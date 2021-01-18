from logging import getLogger
from typing import Any, AnyStr, Sequence, SupportsFloat, SupportsInt, Union

import cv2
import matplotlib.pyplot as plt
import numpy as np

from bikipy.math.geometry import order_parallelogram_corners, expand_parallelogram
from bikipy.utils.video import get_video_data

logger = getLogger(__name__)


class Border:
    def __init__(
        self,
        guiding_image: Union[AnyStr, None] = None,
        semantic_label: Any = None,
        int_label: Union[SupportsInt, None] = None,
    ):
        """
        Parameters
        ----------
        guiding_image: Optional, string
            Label for the border. Useful for manual audition and testing.

        semantic_label: Optional, string
            Label for the border. Useful for manual audition and testing.
        """

        self.guiding_image = guiding_image
        self.semantic_label = semantic_label
        self.int_label = int_label

    def plot(
        self, ax: Any = None, points: Union[Sequence, None] = None, bin: bool = True
    ):
        if not ax:
            fig, ax = plt.subplots()

        if self.guiding_image:
            ax.imshow(cv2.imread(str(self.guiding_image)))

        if points is not None:
            points = np.asanyarray(points)

            if bin:
                histogram, _x_edges, _y_edges = np.histogram2d(
                    *points[np.logical_and(*np.isfinite(points).T)].T, bins=60
                )
                ax.imshow(histogram.T, interpolation="sinc")

            ax.plot(*points.T, ".r-")

        return ax


class PolygonalBorder(Border):
    def __init__(
        self,
        feature_scale: Union[Sequence[SupportsFloat], None] = None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.feature_scale = np.asanyarray(feature_scale) if feature_scale else None

    def confined_coordinates(
        self, coordinates: Sequence, plot: bool = False
    ) -> np.ndarray:
        """
        self.confined_coordinate_indexes to fetch and optionally plot the
        confined coordinates within the respective border

        Parameters
        ----------
        coordinates: Sequence
            Coordinates that will have their confinement tested

        plot: bool
            If True, plot the confined coordinates

        Returns
        -------
        np.ndarray, confined coordinates
        """

        coordinates = np.asanyarray(coordinates)
        confined_coordinates = coordinates[
            self.confined_coordinate_indexes(coordinates)
        ]
        if plot:
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
    ) -> np.ndarray:
        """
        Define sequential border confinements of coordinates

        Parameters
        ----------
        coordinates: Sequence
            Coordinates that will have their confinement tested

        superior_poly_border_instances: Sequence
            PolygonalBorder instances that will have the highest priority
            in case of overlap with respect to confinement

        inferior_poly_border_instances: Sequence
            PolygonalBorder instances that will have the lowest priority
            in case of overlap with respect to confinement

        clean_outliers
            Clear elements that aren't confined to any of the given borders
            as a final action before returning the sequential border presence

        Returns
        -------
        np.ndarray that stores the sequential border presence across frames
        """

        coordinates = np.asanyarray(coordinates)

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
                print(
                    f"Border {border.semantic_label} has coordinate overlap with "
                    f"other borders, {overlap_locations[border.semantic_label].size}"
                )

            presence[confined_coord_booleans_index] = border.int_label

        valid_indexes = np.nonzero(presence)
        if clean_outliers:
            presence = presence[valid_indexes]

        boolean_array = np.full(coordinates.shape[0], False, dtype=np.bool)
        boolean_array[valid_indexes] = True

        return presence, valid_indexes, boolean_array

    @property
    def feat_scaled_sides(self):
        if not self.feature_scale:
            msg = "Feature scale parameters have not been defined in this instance"
            raise AttributeError(msg)
        return self.sides / self.feature_scale


class GenericPolygonalBorder(PolygonalBorder):
    def __init__(
        self,
        sides: Union[Sequence[Sequence[SupportsFloat]], None] = None,
        border_distance: Union[SupportsFloat, None] = None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        if sides:
            self.sides = sides
        if border_distance:
            self.border_distance = border_distance

        self.centroid = np.mean(sides, axis=1)

    @property
    def sides(self):
        return self.__sides

    @sides.setter
    def sides(self, sides: Sequence[Sequence[SupportsFloat]]):
        if (number_of_sides := len(sides)) == 4:
            self.__sides = order_parallelogram_corners(sides)
        else:
            logger.warn(
                f"Number of sides, {number_of_sides}, not supported. The object may "
                f"not work as intended as the sides are not graphed/sorted."
            )
            self.__sides = sides

        self.number_of_sides = number_of_sides

    def __repr__(self):
        return (
            f"\n{self.__class__.__name__}(\n"
            f"    sides={self.sides},\n"
            f"    guiding_image={self.guiding_image},\n"
            f"    label={self.semantic_label}\n"
            ")"
        )

    def __getitem__(self, item):
        return self.sides[item]

    @property
    def order(self):
        return self.sides.shape[0]

    @property
    def borders(self):
        if not self.border_distance:
            msg = "border_distance has to be defined as an object attribute"
            raise AttributeError(msg)

        return expand_parallelogram(self.sides, self.border_distance)

    @staticmethod
    def corner_to_corner_vectors(ordered_corners):
        return (
            *np.diff(ordered_corners, axis=0),
            ordered_corners[0] - ordered_corners[-1],
        )

    @property
    def side_vectors(self):
        return self.corner_to_corner_vectors(self.sides)

    @property
    def border_vectors(self):
        return self.corner_to_corner_vectors(self.borders)

    def plot(self, *args, include_borders: bool = False, **kwargs):
        """
        Plot the sides defined in the object

        Returns
        -------
        matplotlib Axes object with the plot
        """
        ax = super().plot(*args, **kwargs)

        legends = []
        for i in range(len(self.sides)):
            next = 0 if i + 1 == len(self.sides) else i + 1

            side_a = self.sides[i]
            side_b = self.sides[next]
            ax.plot((side_a[0], side_b[0]), (side_a[1], side_b[1]), "o-")

            legend = [self._add_label_to_str(f"side {i}")]

            if include_borders:
                border_a = self.borders[i]
                border_b = self.borders[next]
                ax.plot((border_a[0], border_a[1]), (border_b[0], border_b[1]), "o-")

                legend.append(self._add_label_to_str(f"border {i}"))

            legends.extend(legend)

        plt.legend(legends)
        return ax

    def _add_label_to_str(self, in_string):
        if self.semantic_label:
            return f"{self.semantic_label} {in_string}"
        if self.int_label:
            return f"{self.int_label} {in_string}"

        return in_string

    @staticmethod
    def distance_between_two_borders(border_a, border_b):
        return np.linalg.norm(border_a.centroid - border_b.centroid)

    # Define "corners" (integer) as a class variable for the ginput in from_image(...)
    @classmethod
    def from_image(cls, guiding_image: Any, *args, **kwargs):
        """
        Define the corners of a polygon with a guiding image

        Parameters
        ----------
        guiding_image
            Either path to image or PIL.Image object with opened image inside

        Returns
        -------
        List with pixel coordinates of the polygon corners
        """

        if isinstance(guiding_image, np.ndarray):
            img = guiding_image
        else:
            img = cv2.imread(str(guiding_image))
        plt.imshow(img)

        sides = plt.ginput(n=cls.corners, timeout=0)
        return cls(sides=sides, guiding_image=guiding_image, *args, **kwargs)

    @classmethod
    def from_video(
        cls, video_path: Any, frame_time: AnyStr = "middle", *args, **kwargs
    ):
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
        bikipy.border.polygon.polygon_corners_on_image call with frame from video
        """

        frame, x_res, y_res, _fps = get_video_data(video_path, frame_time)

        return cls.from_image(frame, *args, feature_scale=(x_res, y_res), **kwargs)
