from logging import getLogger
from typing import Any, Sequence, SupportsFloat, SupportsInt, Union

import matplotlib.pyplot as plt
import numpy as np

from bikipy.border.base import PolygonalBorder
from bikipy.border.parallelogram.draw import parallelogram_input
from bikipy.math.geometry import order_parallelogram_corners
from bikipy.math.point_in_polygon import points_in_parallelogram
from bikipy.math.vector import (
    normal_from_line_to_point,
    orthogonal_unit_vector,
    unit_vector,
)

logger = getLogger(__name__)


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
    def sides(self):
        return *self.base, *self.apex

    @sides.setter
    def sides(self, sides: Sequence):
        if (n := len(sides)) != 4:
            msg = f"Parallelogram border has to have 4 sides, got only {n} sides"
            raise ValueError(msg)

        down_left, down_right, up_right, up_left = order_parallelogram_corners(sides)

        self.base = (down_left, down_right)
        self.apex = (up_left, up_right)

        self.sides = (down_left, down_right, up_right, up_left)

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(\n"
            f"    base={self.base.tolist()},\n"
            f"    apex={self.apex.tolist()},\n"
            f'    guiding_image="{self.guiding_image}",\n'
            f'    label="{self.semantic_label}"\n'
            ")"
        )

    @property
    def midline_vector(self):
        return self.apex_mid - self.base_mid

    @property
    def midline_unit(self):
        return unit_vector(self.midline_vector)

    @property
    def midline_unit_orthogonal(self):
        return orthogonal_unit_vector(self.midline_unit)

    @property
    def midline_magnitude(self):
        return np.linalg.norm(self.midline_vector)

    @property
    def close_to_origin_side_vector(self):
        return self.apex[0] - self.base[0]

    @property
    def close_to_origin_side_unit(self):
        return unit_vector(self.close_to_origin_side_vector)

    @property
    def far_from_origin_side_vector(self):
        return self.apex[1] - self.base[1]

    @property
    def far_from_origin_side_unit(self):
        return unit_vector(self.far_from_origin_side_vector)

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

    def plot(self, *args, **kwargs):
        ax = super().plot(*args, **kwargs)

        ax.plot(
            (self.base[0][1], self.base[1][1]),
            "-r",
            (self.apex[0][0], self.apex[1][0]),
            (self.apex[0][1], self.apex[1][1]),
            "-c",
            (self.base[0][0], self.apex[0][0]),
            (self.base[0][1], self.apex[0][1]),
            "-b",
            (self.base[1][0], self.apex[1][0]),
            (self.base[1][1], self.apex[1][1]),
            "-g",
            (self.base_mid[0], self.apex_mid[0]),
            (self.base_mid[1], self.apex_mid[1]),
            "-k",
        )
        plt.legend(("Base", "Apex", "Close Feet", "Far Feet", "Midline"))

        return ax
