from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np
from pydantic import computed_field
from pydantic_numpy.typing import Np1DArrayBool, Np2DArrayFp64

from bikipy.perimeter.polygon.base import BasePolygonPerimeter


class TrianglePerimeter(BasePolygonPerimeter):
    polygon_order = 3

    perimeter_label = "triangle"

    @computed_field  # type: ignore[misc]
    @property
    def base_a(self):
        return self.vertices_in_meters[0]

    @computed_field  # type: ignore[misc]
    @property
    def base_b(self):
        return self.vertices_in_meters[1]

    @computed_field  # type: ignore[misc]
    @property
    def apex(self):
        return self.vertices_in_meters[2]

    def expand(self, perimeter_border_normal_meters: float | Np2DArrayFp64):
        raise NotImplementedError()

    def _compute_confinement_boolean_index(self, coordinates: Np2DArrayFp64) -> Np1DArrayBool:
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

    def compute_filter_by_ray_direction_offset_filter(
        self,
        op_label: str,
        ray_start_points: Np2DArrayFp64,
        ray_travel_direction_points: Np2DArrayFp64,
        max_radians: float,
        angular_resolution: int = 400,
        extra_ax: Optional[plt.Axes] = None,
    ) -> tuple[Np1DArrayBool, dict[str, Any]]:
        method = (
            self.circle.filter_by_ray_direction_offset_filter
            if self.equilateral
            else super().filter_by_ray_direction_offset_filter
        )
        return method(
            self.__class__.__name__,
            ray_start_points=ray_start_points,
            ray_travel_direction_points=ray_travel_direction_points,
            max_radians=max_radians,
            extra_ax=extra_ax,
        )
