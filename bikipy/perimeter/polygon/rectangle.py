from logging import getLogger
from typing import ClassVar, Optional

import matplotlib.pyplot as plt

from bikipy.core.typing import NDArrayFp64
from bikipy.perimeter.polygon.base import BasePolygonPerimeter
from bikipy.utils.math.geometry import expand_rectangle
from bikipy.utils.plotting import generic_inspection_finalization

logger = getLogger(__name__)


class RectanglePerimeter(BasePolygonPerimeter):
    polygon_order = 4

    def expand(self, perimeter_border_normal_pixels: float | NDArrayFp64) -> "RectanglePerimeter":
        result = self.__class__(
            vertices_in_pixels=expand_rectangle(
                self.vertices_in_pixels,
                perimeter_border_normal_pixels,
            ),
            manual_video=self.video,
            label=f"border_{self.label}",
        )

        if self.inspect_arg:
            fig, ax = plt.subplots()
            ax.set_title("ExpandPerimeter")

            self.plot_perimeter(manual_ax=ax, label="Original")
            result.plot_perimeter(manual_ax=ax, label="Expanded")

            ax.legend()

            generic_inspection_finalization(self.expand_inspect_arg)

        return result
