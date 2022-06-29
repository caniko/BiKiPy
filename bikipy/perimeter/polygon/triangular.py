from typing import ClassVar, Optional

import numpy as np

from bikipy.core.typing import NDArrayFp64
from bikipy.perimeter.polygon.base import PolygonPerimeter


class TriangularPerimeter(PolygonPerimeter):
    polygon_order: ClassVar[Optional[int]] = 3

    @property
    def base_a(self):
        return self.vertices_in_meters[0]

    @property
    def base_b(self):
        return self.vertices_in_meters[1]

    @property
    def apex(self):
        return self.vertices_in_meters[2]

    def expand(self, perimeter_border_normal_meters: float | NDArrayFp64):
        pass

    def coordinate_confinement_boolean_index(self, coordinates: NDArrayFp64) -> NDArrayFp64:
        """
        indices of the coordinates that are inside the respective perimeter

        Parameters
        ----------
        coordinates
            Sequence of coordinates

        Returns
        -------
        NDArrayFp64 of all the indices
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
