from functools import cached_property
from logging import getLogger
from typing import Any, ClassVar, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from bikipy.math.point_in_polygon import (
    points_in_parallelogram,
    parallel_point_in_polygon,
)
from bikipy.math.vector import (
    normal_from_line_to_point,
    orthogonal_unit_vector,
    unit_vector,
)
from bikipy.perimeter.polygon.base import PolygonPerimeter
from bikipy.perimeter.polygon.parallelogram.draw import parallelogram_input

logger = getLogger(__name__)


class ParallelogramPerimeter(PolygonPerimeter):
    _polygon_order: ClassVar[Optional[int]] = 4

    @classmethod
    def from_image(cls, inspect_image: Any, *args, **kwargs):
        base, apex = parallelogram_input(inspect_image)
        return cls(corners=np.array((*base, *apex)), inspect_image=inspect_image)

    @staticmethod
    def midpoint(
        close_corner: Sequence[float], far_corner: Sequence[float]
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
        return close_corner + (far_corner - close_corner) / 2.0

    @staticmethod
    def sort_vectors(vectors: Sequence) -> np.ndarray:
        vectors = np.asarray(vectors)

        assert vectors.shape == (2, 2), (
            f"Vector set must define the endpoints of a side,"
            f"the shape must therefore be (2, 2) and not {vectors.shape}"
        )

        vector_norms = np.argsort(np.linalg.norm(vectors, axis=1))
        return vectors[vector_norms]

    @property
    def down_left(self):
        return self.corners[0]

    @property
    def down_right(self):
        return self.corners[1]

    @property
    def up_right(self):
        return self.corners[2]

    @property
    def up_left(self):
        return self.corners[3]

    @property
    def base(self):
        return self.corners[:2]

    @cached_property
    def base_vector(self):
        return np.diff(self.base)

    @cached_property
    def base_mindpoint(self):
        return self.midpoint(*self.base)

    @property
    def apex(self):
        return self.corners[2:]

    @cached_property
    def apex_vector(self):
        return np.diff(self.apex)

    @cached_property
    def apex_mindpoint(self):
        return self.midpoint(*self.apex)

    @cached_property
    def midline_vector(self):
        return self.apex_mindpoint - self.base_mindpoint

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
            lambda x: normal_from_line_to_point(
                self.midline_unit, self.base_mindpoint, x
            ),
            1,
            coordinates,
        )

        return np.squeeze(np.hsplit(magnitudes, 2))

    def coordinate_confinement_boolean_index(
        self, coordinates: NDArray, *args, **kwargs
    ):
        return parallel_point_in_polygon(coordinates, self.corners)
        # return points_in_parallelogram(
        #     self.corners[3],
        #     self.corners[0],
        #     self.corners[2],
        #     np.asarray(coordinates),
        #     *args,
        #     **kwargs,
        # )

    @classmethod
    def many(
        cls,
        inspect_image: Any,
        n: int,
        object_kwargs: Optional[Sequence] = None,
    ):
        return [
            cls(inspect_image=inspect_image, **object_kwargs[i]) for i in range(int(n))
        ]

    def plot_parallelogram_labels(self, *args, **kwargs):
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
            (self.base_mindpoint[0], self.apex_mindpoint[0]),
            (self.base_mindpoint[1], self.apex_mindpoint[1]),
            "-k",
        )
        plt.legend(
            ("Base", "Apex", "Close Feet", "Far Feet", "Midline"),
            bbox_to_anchor=(1.04, 0.5),
            loc="center left",
        )

        return ax
