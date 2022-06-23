from functools import cached_property
from logging import getLogger
from typing import Any, ClassVar, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np

from bikipy.core.typing import NDArrayFp64
from bikipy.perimeter.polygon.base import PolygonPerimeter
from bikipy.utils.math.geometry import expand_rectangle
from bikipy.utils.math.point_in_polygon import parallel_point_in_polygon
from bikipy.utils.math.vector import (
    normal_from_line_to_point,
    orthogonal_unit_vector,
    unit_vector,
)

logger = getLogger(__name__)


class RectanglePerimeter(PolygonPerimeter):
    polygon_order: ClassVar[Optional[int]] = 4

    def expand(self, perimeter_border_normal_meters: float | NDArrayFp64) -> "RectanglePerimeter":
        return self.__class__(
            vertices_in_pixels=expand_rectangle(self.vertices_in_pixels, perimeter_border_normal_meters),
            manual_video=self.video
        )
