from logging import getLogger
from typing import ClassVar, Optional

from bikipy.core.typing import NDArrayFp64
from bikipy.perimeter.polygon.base import PolygonPerimeter
from bikipy.utils.math.geometry import expand_rectangle

logger = getLogger(__name__)


class RectanglePerimeter(PolygonPerimeter):
    polygon_order: ClassVar[Optional[int]] = 4

    def expand(self, perimeter_border_normal_meters: float | NDArrayFp64) -> "RectanglePerimeter":
        return self.__class__(
            vertices_in_pixels=expand_rectangle(self.vertices_in_pixels, perimeter_border_normal_meters),
            manual_video=self.video,
            label=f"border_{self.label}",
        )
