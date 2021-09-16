from collections.abc import Sequence
from functools import cached_property
from typing import Union

import numpy as np

from bikipy.perimeter.base import PolygonalPerimeter


class TriangularPerimeter(PolygonalPerimeter):
    corners = 3

    def __init__(
        self,
        base_a: Union[Sequence[float], None] = None,
        base_b: Union[Sequence[float], None] = None,
        apex: Union[Sequence[float], None] = None,
        perimeter_corners: Union[Sequence[Sequence[float]]] = None,
        *args,
        **kwargs,
    ):
        """

        Parameters
        ----------
        base_a
            Coordinates of one of the perimeter_corners that denote the base of the triangle
        base_b
            Coordinates of one of the perimeter_corners that denote the base of the triangle
        apex
            Coordinates of one of the perimeter_corners that denote the apex of the triangle
        kwargs
        """

        if perimeter_corners:
            base_a, base_b, apex = perimeter_corners
        elif not (base_a and base_b and apex):
            msg = "perimeter_corners or (base_a, base_b, apex) has to be defined"
            raise ValueError(msg)

        self.base_a = np.asarray(base_a)
        self.base_b = np.asarray(base_b)
        self.apex = np.asarray(apex)

        super().__init__(
            perimeter_corners=(self.base_a, self.base_b, self.apex), *args, **kwargs
        )

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(\n"
            f"    base_a={self.base_a.tolist()},\n"
            f"    base_b={self.base_b.tolist()},\n"
            f"    apex={self.apex.tolist()},\n"
            f'    inspect_image="{self.inspect_image}",\n'
            f'    label="{self.semantic_label}"\n'
            ")"
        )

    def coordinate_confinement_boolean_index(
        self, coordinates: np.ndarray
    ) -> np.ndarray:
        """
        indices of the coordinates that are inside the respective perimeter

        Parameters
        ----------
        coordinates
            Sequence of coordinates

        Returns
        -------
        np.ndarray of all the indices
        """
        coord_x_comp, coord_y_comp = np.asarray(coordinates).T

        c1 = (self.base_b[0] - self.base_a[0]) * (coord_y_comp - self.base_a[1]) - (
            self.base_b[1] - self.base_a[1]
        ) * (coord_x_comp - self.base_a[0])
        c2 = (self.apex[0] - self.base_b[0]) * (coord_y_comp - self.base_b[1]) - (
            self.apex[1] - self.base_b[1]
        ) * (coord_x_comp - self.base_b[0])
        c3 = (self.base_a[0] - self.apex[0]) * (coord_y_comp - self.apex[1]) - (
            self.base_a[1] - self.apex[1]
        ) * (coord_x_comp - self.apex[0])

        return np.logical_or(
            np.logical_and(c1 > 0, np.logical_and(c2 > 0, c3 > 0)),
            np.logical_and(c1 < 0, np.logical_and(c2 < 0, c3 < 0)),
        )

    @cached_property
    def edge_midpoints(self):
        return (
            self.perimeter_corners
            + np.diff(self.perimeter_corners, append=self.base_a) / 2.0
        )
