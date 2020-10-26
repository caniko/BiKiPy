from typing import Union, Any, AnyStr, SupportsFloat, SupportsInt, Sequence

import matplotlib.pyplot as plt
import numpy as np

from bikipy.math.vector import (
    unit_vector,
    orthogonal_vector,
    normal_from_line_to_point,
)

from bikipy.border.parallelogram.draw import parallelogram_input
from bikipy.math.vector import find_intersection_between_two_vectors
from bikipy.math.point_in_polygon import points_in_parallelogram
from bikipy.border.base import PolygonalBorder


class ParallelogramBorder(PolygonalBorder):
    def __init__(
        self,
        base: Union[Sequence[SupportsFloat], None] = None,
        apex: Union[Sequence[SupportsFloat], None] = None,
        **kwargs,
    ):
        """
        Parameters
        ----------
        base: Sequence
            The coordinates of the sides of the base of the parallelogram
        apex: Sequence
            The coordinates of the sides of the apex of the parallelogram
        guiding_image: Path to image
            Image used for annotating base and apex; apex and base cannot be defined
            if guiding_image is defined
        """
        super().__init__(**kwargs)

        if not base and not apex:
            if not self.guiding_image:
                msg = "Image cannot be defined when base and apex are defined"
                raise AttributeError(msg)
            self.base, self.apex = parallelogram_input(self.guiding_image)
        else:
            self.base, self.apex = base, apex

    @staticmethod
    def midpoint(
        close_corner: Sequence[SupportsFloat], far_corner: Sequence[SupportsFloat]
    ) -> np.ndarray:
        """
        Find the midpoint of the parallelogram

        Parameters
        ----------
        close_corner
        far_corner

        Returns
        -------

        """
        close_corner, far_corner = (
            np.asanyarray(close_corner),
            np.asanyarray(far_corner),
        )
        return close_corner + (far_corner - close_corner) / 2.0

    @staticmethod
    def sort_vectors(vectors: Sequence) -> np.ndarray:
        vectors = np.asanyarray(vectors)

        assert vectors.shape == (2, 2), (
            f"Vector set must define the endpoints of a side,"
            f"the shape must therefore be (2, 2) and not {vectors.shape}"
        )

        vector_norms = np.argsort(np.linalg.norm(vectors, axis=1))
        return vectors[vector_norms]

    @property
    def base(self):
        return self.__base

    @base.setter
    def base(self, base: Sequence):
        base = np.asanyarray(base)
        self.__base = self.sort_vectors(base)

        # self.base_mid = self.base[0] + (self.base[1] - self.base[0]) / 2
        self.base_mid = self.midpoint(*self.__base)
        self.base_vector = base[1] - base[0]

    @property
    def apex(self):
        return self.__apex

    @apex.setter
    def apex(self, apex: Sequence):
        apex = np.asanyarray(apex)
        self.__apex = self.sort_vectors(apex)

        # self.apex_mid = self.apex[0] + (self.apex[1] - self.apex[0]) / 2
        self.apex_mid = self.midpoint(*self.__apex)
        self.apex_vector = apex[1] - apex[0]

    @property
    def midline_vector(self):
        return self.apex_mid - self.base_mid

    @property
    def midline_unit(self):
        return unit_vector(self.midline_vector)

    @property
    def midline_unit_orthogonal(self):
        return orthogonal_vector(self.midline_unit)

    @property
    def midline_magnitude(self):
        return np.linalg.norm(self.midline_vector)

    @property
    def close_feet_vector(self):
        return self.apex[0] - self.base[0]

    @property
    def close_feet_unit(self):
        return unit_vector(self.close_feet_vector)

    @property
    def far_feet_vector(self):
        return self.apex[1] - self.base[1]

    @property
    def far_feet_unit(self):
        return unit_vector(self.far_feet_vector)

    def base_midpoint_coordinate_unit_vector_magnitudes(
        self, coordinates: Sequence
    ) -> np.ndarray:
        """
        Generate the magnitude of the line segment that goes from origin
        to the defined coordinate

        Parameters
        ----------
        coordinates
            Sequence of coordinates

        Returns
        -------
        np.ndarray, where 1st row is base to midpoint apex magnitudes;
        2nd row is midpoint apex to coordinate magnitudes
        """

        coordinates = np.asanyarray(coordinates)
        magnitudes = np.apply_along_axis(
            lambda x: normal_from_line_to_point(self.midline_unit, self.base_mid, x),
            1,
            coordinates,
        )

        return np.squeeze(np.hsplit(magnitudes, 2))

    def confined_coordinate_indexes(self, coordinates: Sequence):
        coordinates = np.asanyarray(coordinates)

        return points_in_parallelogram(
            self.base[0], self.apex[0], self.base[1], coordinates
        )

    @classmethod
    def many(
        cls,
        guiding_image: Any,
        n: SupportsInt,
        object_kwargs: Union[Sequence, None] = None,
    ):
        return [
            cls(guiding_image=guiding_image, **object_kwargs[i]) for i in range(int(n))
        ]

    def plot(self, points: Union[Sequence, None] = None, show: bool = True):
        fig, ax = plt.subplots()
        ax.plot(
            # Base
            (self.base[0][0], self.base[1][0]),
            (self.base[0][1], self.base[1][1]),
            "-r",
            # Apex
            (self.apex[0][0], self.apex[1][0]),
            (self.apex[0][1], self.apex[1][1]),
            "-c",
            # Close feet
            (self.base[0][0], self.apex[0][0]),
            (self.base[0][1], self.apex[0][1]),
            "-b",
            # Far feet
            (self.base[1][0], self.apex[1][0]),
            (self.base[1][1], self.apex[1][1]),
            "-g",
            # Midline
            (self.base_mid[0], self.apex_mid[0]),
            (self.base_mid[1], self.apex_mid[1]),
            "-k",
        )
        plt.legend(("Base", "Apex", "Close Feet", "Far Feet", "Midline"))
        if points is not None:
            points = np.asanyarray(points)
            ax.scatter(points.T[0], points.T[1], marker=".")
        if show:
            self.plt_show(ax)
        return fig, ax


class GradientBorder(ParallelogramBorder):
    def __init__(
        self,
        gradient_range: Sequence = (0.5, 1.0),
        **kwargs,
    ):
        """

        Parameters
        ----------
        base: Sequence
            The coordinates of the sides of the base of the parallelogram
        apex: Sequence
            The coordinates of the sides of the apex of the parallelogram
        gradient_range: Sequence
            The range of the gradient; (starting weight, end weight)
        guiding_image: Path to image
            Image used for annotating base and apex; apex and base cannot be defined
            if guiding_image is defined
        """
        super().__init__(**kwargs)

        self.gradient_range = gradient_range

    @property
    def gradient_range(self):
        return self.__gradient_range

    @gradient_range.setter
    def gradient_range(self, gradient_range: Sequence):
        gradient_min, gradient_max = tuple(gradient_range)
        if gradient_min >= gradient_max:
            msg = "gradient_range has to be less than self.gradient_max"
            raise ValueError(msg)

        if not 0.0 <= gradient_min <= 1.0 or not 0.0 <= gradient_max <= 1.0:
            msg = "gradient range must be between 0 and 1.0"
            raise ValueError(msg)

        self.__gradient_range = gradient_range

    @property
    def gradient(self):
        return np.linspace(*self.gradient_range, 100000)

    def confined_coordinate_indexes(self, coordinates: Sequence) -> np.ndarray:
        """

        Parameters
        ----------
        coordinates
            Sequence of coordinates

        Returns
        -------

        """
        coordinates = np.asanyarray(coordinates)
        (
            base_to_mid_apex_magnitudes,
            mid_apex_to_coordinate_magnitudes,
        ) = self.base_midpoint_coordinate_unit_vector_magnitudes(coordinates)

        valid_coordinates_boolean_indexes = base_to_mid_apex_magnitudes <= 0
        east_of_midpoint_booleans = mid_apex_to_coordinate_magnitudes <= 0

        valid_magnitudes = base_to_mid_apex_magnitudes[
            valid_coordinates_boolean_indexes
        ]
        expanded_valid_magnitudes = np.expand_dims(valid_magnitudes, 0).T

        line_segment_apex = (
            self.base_mid - expanded_valid_magnitudes * self.midline_unit
            if self.base_mid[0] > self.apex_mid[0]
            else self.base_mid + expanded_valid_magnitudes * self.midline_unit
        )

        compute = line_segment_apex.copy()
        east_of_midpoint_booleans = east_of_midpoint_booleans[
            valid_coordinates_boolean_indexes
        ]
        west_of_midpoint_booleans = np.logical_not(east_of_midpoint_booleans)
        midline_to_coordinates_norm = np.linalg.norm(
            coordinates[valid_coordinates_boolean_indexes.T[0]] - line_segment_apex,
            axis=1,
        )

        if self.base_mid[1] >= self.apex_mid[1]:
            compute[east_of_midpoint_booleans] = np.apply_along_axis(
                lambda x: find_intersection_between_two_vectors(
                    self.midline_unit_orthogonal, self.far_feet_unit, x, self.base[1]
                ),
                1,
                compute[east_of_midpoint_booleans],
            )
            compute[west_of_midpoint_booleans] = np.apply_along_axis(
                lambda x: find_intersection_between_two_vectors(
                    self.midline_unit_orthogonal, self.close_feet_unit, x, self.base[0]
                ),
                1,
                compute[west_of_midpoint_booleans],
            )

            compute = compute + line_segment_apex

            compute = np.linalg.norm(compute, axis=1)
            compute[east_of_midpoint_booleans] = (
                compute[east_of_midpoint_booleans]
                >= midline_to_coordinates_norm[east_of_midpoint_booleans]
            )
            compute[west_of_midpoint_booleans] = (
                compute[west_of_midpoint_booleans]
                <= midline_to_coordinates_norm[west_of_midpoint_booleans]
            )

        else:
            compute[west_of_midpoint_booleans] = np.apply_along_axis(
                lambda x: find_intersection_between_two_vectors(
                    self.midline_unit_orthogonal, self.far_feet_unit, x, self.base[1]
                ),
                1,
                compute[west_of_midpoint_booleans],
            )
            compute[east_of_midpoint_booleans] = np.apply_along_axis(
                lambda x: find_intersection_between_two_vectors(
                    self.midline_unit_orthogonal, self.close_feet_unit, x, self.base[0]
                ),
                1,
                compute[east_of_midpoint_booleans],
            )
            compute = np.linalg.norm(line_segment_apex - compute, axis=1)
            compute[east_of_midpoint_booleans] = (
                compute[east_of_midpoint_booleans]
                <= midline_to_coordinates_norm[east_of_midpoint_booleans]
            )
            compute[west_of_midpoint_booleans] = (
                compute[west_of_midpoint_booleans]
                >= midline_to_coordinates_norm[west_of_midpoint_booleans]
            )

        line_to_coord_segment = line_segment_apex + np.expand_dims(
            mid_apex_to_coordinate_magnitudes[valid_coordinates_boolean_indexes], 0
        ).T * orthogonal_vector(self.midline_unit)

        fig, ax = self.plot(show=False)
        ax.scatter(
            coordinates[valid_coordinates_boolean_indexes.T[0]].T[0],
            coordinates[valid_coordinates_boolean_indexes.T[0]].T[1],
            marker="x",
        )
        self.plt_show(ax)

        return valid_coordinates_boolean_indexes.T[0]

    def weight_coordinates(self, coordinates: Sequence, plot: AnyStr = "scatter"):
        coordinates = np.asanyarray(coordinates)
        (
            base_to_mid_apex_magnitudes,
            mid_apex_to_coordinate_magnitudes,
        ) = self.line_segment_magnitudes(coordinates)

        valid_coordinates_boolean_indexes = (
            base_to_mid_apex_magnitudes < 0
            if self.base_mid[0] > self.apex_mid[0]
            else base_to_mid_apex_magnitudes > 0
        )
        weights = np.abs(valid_magnitudes / self.midline_magnitude)

        if plot:
            fig, ax = self.plot(show=False)
            line_segment_apex = (
                self.base_mid - valid_magnitudes * self.midline_unit
                if self.base_mid[0] > self.apex_mid[0]
                else self.base_mid + valid_magnitudes * self.midline_unit
            )

            if plot == "scatter":
                ax.scatter(line_segment_apex.T[0], line_segment_apex.T[1])
            elif plot == "normal":
                ax.plot(
                    (line_segment_apex.T[0], coordinates.T[0]),
                    (line_segment_apex.T[1], coordinates.T[1]),
                )

            self.plt_show(ax)

        return weights
