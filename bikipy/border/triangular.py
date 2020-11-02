from typing import Union, SupportsFloat, Sequence

import matplotlib.pyplot as plt
import numpy as np

from bikipy.border.base import GenericPolygonalBorder


class TriangularBorder(GenericPolygonalBorder):
    corners = 3

    def __init__(
        self,
        base_a: Sequence[SupportsFloat],
        base_b: Sequence[SupportsFloat],
        apex: Sequence[SupportsFloat],
        *args,
        **kwargs,
    ):
        """

        Parameters
        ----------
        base_a
            Coordinates of one of the sides that denote the base of the triangle
        base_b
            Coordinates of one of the sides that denote the base of the triangle
        apex
            Coordinates of one of the sides that denote the apex of the triangle
        kwargs
        """

        self.base_a = np.asanyarray(base_a)
        self.base_b = np.asanyarray(base_b)
        self.apex = np.asanyarray(apex)

        sides = (self.base_a, self.base_b, self.apex)

        super().__init__(*args, sides=sides, **kwargs)

    def confined_coordinate_indexes(self, coordinates: Sequence) -> np.ndarray:
        """
        Indexes of the coordinates that are inside the respective border

        Parameters
        ----------
        coordinates
            Sequence of coordinates

        Returns
        -------
        np.ndarray of all the indexes
        """
        coord_x_comp, coord_y_comp = np.asanyarray(coordinates).T

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

    def plot(self, points: Union[Sequence, None] = None, show: bool = True):
        fig, ax = plt.subplots()
        ax.plot(
            # Base
            (self.base_a[0], self.base_b[0]),
            (self.base_a[1], self.base_b[1]),
            "-r",
            # base_b -> apex
            (self.base_b[0], self.apex[0]),
            (self.base_b[1], self.apex[1]),
            "-c",
            # base_a -> apex
            (self.base_a[0], self.apex[0]),
            (self.base_a[1], self.apex[1]),
            "-b",
        )

        if points is not None:
            points = np.asanyarray(points)
            ax.scatter(points.T[0], points.T[1], marker=".")
        if show:
            self.plt_show(ax)

        return fig, ax
