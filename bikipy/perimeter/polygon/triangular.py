from collections.abc import Sequence
from typing import ClassVar, Optional, Union

import numpy as np
from pydantic_numpy import NDArray

from bikipy.perimeter.polygon.base import PolygonPerimeter


class TriangularPerimeter(PolygonPerimeter):
    _polygon_order: ClassVar[Optional[int]] = 3

    @property
    def base_a(self):
        return self.corners[0]

    @property
    def base_b(self):
        return self.corners[1]

    @property
    def apex(self):
        return self.corners[2]

    def coordinate_confinement_boolean_index(self, coordinates: NDArray) -> NDArray:
        """
        indices of the coordinates that are inside the respective perimeter

        Parameters
        ----------
        coordinates
            Sequence of coordinates

        Returns
        -------
        NDArray of all the indices
        """
        coord_x_comp, coord_y_comp = np.asarray(coordinates).T

        c1 = (self.base_b[0] - self.base_a[0]) * (coord_y_comp - self.base_a[1]) - (self.base_b[1] - self.base_a[1]) * (
            coord_x_comp - self.base_a[0]
        )
        c2 = (self.apex[0] - self.base_b[0]) * (coord_y_comp - self.base_b[1]) - (self.apex[1] - self.base_b[1]) * (
            coord_x_comp - self.base_b[0]
        )
        c3 = (self.base_a[0] - self.apex[0]) * (coord_y_comp - self.apex[1]) - (self.base_a[1] - self.apex[1]) * (
            coord_x_comp - self.apex[0]
        )

        return np.logical_or(
            np.logical_and(c1 > 0, np.logical_and(c2 > 0, c3 > 0)),
            np.logical_and(c1 < 0, np.logical_and(c2 < 0, c3 < 0)),
        )
