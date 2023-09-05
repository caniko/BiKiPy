from logging import getLogger
from typing import Literal

import numpy as np
from matplotlib.axes import Axes
from pydantic import computed_field
from pydantic_numpy.typing import NpNDArrayFp64

from bikipy.math import expand_rectangle
from bikipy.perimeter.polygon.base import BasePolygonPerimeter

logger = getLogger(__name__)


class RectanglePerimeter(BasePolygonPerimeter):
    derived_meters_per_pixel_source: Literal["diagonal", "side", None] = None

    polygon_order = 4

    perimeter_label = "rectangle"

    @computed_field
    @property
    def derived_meters_per_pixel(self) -> float:
        if result := super().derived_meters_per_pixel:
            return result
        if self.derived_meters_per_pixel_source == "diagonal":
            return self.derived_meters_per_pixel_source_metric_length / np.linalg.norm(
                self.vertices_in_pixels.edge_lengths[0] - self.vertices_in_pixels.edge_lengths[2]
            )

    def expand(
        self, perimeter_border_normal_pixels: float | NpNDArrayFp64, ax: Axes = None, **inspect_kwargs
    ) -> "RectanglePerimeter":
        expanded_vertices = expand_rectangle(
            self.vertices_in_pixels,
            x_offset=perimeter_border_normal_pixels,
            y_offset=perimeter_border_normal_pixels,
            inspection_fig_output_path=self.inspection_fig_output_path,
        )

        if any(np.any(expanded_vertex > self.video.recording_resolution) for expanded_vertex in expanded_vertices):
            msg = "The expanded vertex is out of bounds with respect to the video"
            raise ValueError(msg)

        return self.__class__(vertices_in_pixels=expanded_vertices, manual_video=self.video, label=self.label)
