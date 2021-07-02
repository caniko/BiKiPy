from functools import cached_property
from logging import getLogger
from typing import Any, Sequence, SupportsFloat, SupportsInt, Union

import matplotlib.pyplot as plt
import numpy as np

from bikipy.math.geometry import order_parallelogram_corners
from bikipy.math.point_in_polygon import points_in_parallelogram
from bikipy.math.vector import (
    normal_from_line_to_point,
    orthogonal_unit_vector,
    unit_vector,
)
from bikipy.perimeter.base import PolygonalPerimeter
from bikipy.perimeter.parallelogram.draw import parallelogram_input

logger = getLogger(__name__)


class ParallelogramPerimeter(PolygonalPerimeter):
    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        if (n := len(self.perimeter_corners)) != 4:
            msg = (
                f"Parallelogram is a polygon in the 4th order, the current polygon"
                f"is in the {n} order"
            )
            raise ValueError(msg)

        self.perimeter_corners = order_parallelogram_corners(
            self.perimeter_corners
        )
        self.down_left, self.down_right, self.up_right, self.up_left = self.perimeter_corners

        self.base = (self.down_left, self.down_right)
        # self.base_mid = self.base[0] + (self.base[1] - self.base[0]) / 2.
        self.base_mid = self.midpoint(*self.base)
        self.base_vector = self.down_right - self.down_left

        self.apex = (self.up_left, self.up_right)
        # self.apex_mid = self.apex[0] + (self.apex[1] - self.apex[0]) / 2.
        self.apex_mid = self.midpoint(*self.apex)
        self.apex_vector = self.up_right - self.up_left

    def __repr__(self):
        return super().__repr__() + (
            f"\n\tbase={self.base},\n\t" f"apex={self.apex},\n\t"
        )

    @classmethod
    def from_image(cls, inspect_image: Any, n: int, *args, **kwargs):
        base, apex = parallelogram_input(inspect_image)
        return cls(base, apex, inspect_image=inspect_image)

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
            np.asarray(close_corner),
            np.asarray(far_corner),
        )
        return close_corner + (far_corner - close_corner) / 20

    @staticmethod
    def sort_vectors(vectors: Sequence) -> np.ndarray:
        vectors = np.asarray(vectors)

        assert vectors.shape == (2, 2), (
            f"Vector set must define the endpoints of a side,"
            f"the shape must therefore be (2, 2) and not {vectors.shape}"
        )

        vector_norms = np.argsort(np.linalg.norm(vectors, axis=1))
        return vectors[vector_norms]

    @cached_property
    def midline_vector(self):
        return self.apex_mid - self.base_mid

    @cached_property
    def midline_unit(self):
        return unit_vector(self.midline_vector)

    @cached_property
    def midline_unit_orthogonal(self):
        return orthogonal_unit_vector(self.midline_unit)

    @cached_property
    def midline_magnitude(self):
        return np.linalg.norm(self.midline_vector)

    @cached_property
    def close_to_origin_side_vector(self):
        return self.apex[0] - self.base[0]

    @cached_property
    def close_to_origin_side_unit(self):
        return unit_vector(self.close_to_origin_side_vector)

    @cached_property
    def far_from_origin_side_vector(self):
        return self.apex[1] - self.base[1]

    @cached_property
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

        coordinates = np.asarray(coordinates)
        magnitudes = np.apply_along_axis(
            lambda x: normal_from_line_to_point(self.midline_unit, self.base_mid, x),
            1,
            coordinates,
        )

        return np.squeeze(np.hsplit(magnitudes, 2))

    def polygon_contained_coordinates_boolean_index(self, coordinates: Sequence):
        coordinates = np.asarray(coordinates)

        return points_in_parallelogram(
            self.base[0], self.apex[0], self.base[1], coordinates
        )

    @classmethod
    def many(
        cls,
        inspect_image: Any,
        n: SupportsInt,
        object_kwargs: Union[Sequence, None] = None,
    ):
        return [
            cls(inspect_image=inspect_image, **object_kwargs[i]) for i in range(int(n))
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
        plt.legend(
            ("Base", "Apex", "Close Feet", "Far Feet", "Midline"),
            bbox_to_anchor=(1.04, 0.5),
            loc="center left",
        )

        return ax
