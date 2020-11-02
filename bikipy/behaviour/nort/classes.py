from typing import Union, SupportsFloat, Sequence

import numpy as np

from bikipy.border.base import GenericPolygonalBorder
from bikipy.math.vector import unit_vector


class NortObject(GenericPolygonalBorder):
    corners = 0

    def __init__(
        self,
        *args,
        border_distance: Union[SupportsFloat, None] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.border_distance = border_distance

    @property
    def borders(self):
        if not self.border_distance:
            msg = "border_distance has to be defined as an object attribute"
            raise AttributeError(msg)

        diagonal_unit_2_0 = unit_vector(self.sides[0] - self.sides[2])
        diagonal_unit_3_1 = unit_vector(self.sides[1] - self.sides[3])

        return np.array(
            (
                self.sides[0] + diagonal_unit_2_0 * self.border_distance,
                self.sides[1] + diagonal_unit_3_1 * self.border_distance,
                self.sides[2] - diagonal_unit_2_0 * self.border_distance,
                self.sides[3] - diagonal_unit_3_1 * self.border_distance,
            )
        )

    def plot_borders(self, points: Union[Sequence, None] = None, show: bool = True):
        """

        Parameters
        ----------
        points
            User defined coordinates that will be plotted alongside the object

        show
            If True, the plot will be shown through plt.show()

        Returns
        -------

        """

        fig, ax = self.plot(points)
        for i in range(len(self.borders) - 1):
            border_a = self.borders[i]
            border_b = self.borders[i+1]

            ax.plot((border_a[0], border_a[1]), (border_b[0], border_b[1]))

        if show:
            self.plt_show(ax)

        return fig, ax
