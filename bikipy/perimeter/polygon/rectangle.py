from logging import getLogger
from typing import Literal

import numpy as np
from matplotlib.axes import Axes
from pydantic_numpy.dtype import NDArrayFp64

from bikipy.perimeter.polygon.base import BasePolygonPerimeter
from bikipy.utils.math.geometry import expand_rectangle
from bikipy.utils.plot.inspect import generic_inspection_finalization

logger = getLogger(__name__)


class RectanglePerimeter(BasePolygonPerimeter):
    derived_meters_per_pixel_source: Literal["diagonal", "side", None] = None

    polygon_order = 4

    class_inspect_directory_name = "rectangle"

    @property
    def derived_meters_per_pixel(self) -> float:
        if upstream := super().derived_meters_per_pixel:
            return upstream
        if self.derived_meters_per_pixel_source == "diagonal":
            return self.derived_meters_per_pixel_source_metric_length / np.linalg.norm(
                self.vertices_in_pixels.edge_lengths[0] - self.vertices_in_pixels.edge_lengths[2]
            )

    def expand(
        self, perimeter_border_normal_pixels: float | NDArrayFp64, ax: Axes = None, **inspect_kwargs
    ) -> "RectanglePerimeter":
        result = self.__class__(
            vertices_in_pixels=expand_rectangle(
                self.vertices_in_pixels,
                perimeter_border_normal_pixels,
            ),
            manual_video=self.video,
            label=f"border_{self.label}",
        )

        if self.inspect_arg:
            if ax is None:
                fig, ax = self.video.subplot()
                ax.set_title("ExpandPerimeter")

            self.plot_perimeter(manual_ax=ax, label="Original")
            result.plot_perimeter(manual_ax=ax, label="Expanded")

            ax.legend()

            generic_inspection_finalization(self.class_inspect_arg / "expand", **inspect_kwargs)

        return result
