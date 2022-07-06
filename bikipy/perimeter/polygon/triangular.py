from typing import ClassVar, Optional, Any

import numpy as np

from bikipy.core.typing import NDArrayBool, NDArrayFp64
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

    def confined_coordinate_boolean_index(self, coordinates: NDArrayFp64) -> NDArrayFp64:
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
            (c1 > 0.0) & (c2 > 0.0) & (c3 > 0.0),
            (c1 < 0.0) & (c2 < 0.0) & (c3 < 0.0),
        )

    def gaze_direction_filter(
        self,
        gaze_travel_direction_point: NDArrayFp64,
        gaze_start_point: NDArrayFp64,
        max_radians: float,
        manual_ax: Any = None,
        **kwargs
    ) -> NDArrayBool:
        if self.equilateral:
            return self.circle.gaze_direction_filter_circle_triangle(
                gaze_travel_direction_point=gaze_travel_direction_point,
                gaze_start_point=gaze_start_point,
                max_radians=max_radians,
                manual_ax=manual_ax,
                **kwargs
            )
        return super().gaze_direction_filter(
            gaze_travel_direction_point=gaze_travel_direction_point,
            gaze_start_point=gaze_start_point,
            max_radians=max_radians,
            manual_ax=manual_ax,
            **kwargs
        )
