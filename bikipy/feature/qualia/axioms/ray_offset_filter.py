from abc import ABC
from functools import cached_property
from typing import Any, Optional

from matplotlib.axes import Axes
from pydantic import computed_field
from pydantic_numpy.typing import Np1DArrayBool, Np2DArrayFp64

from bikipy.core.compute import AbstractComputePerimeterBooleanIndex
from bikipy.core.video import VideoMetadata
from bikipy.feature.tolerance.single import single_node_tolerance_model
from bikipy.perimeter.base import BasePerimeter
from bikipy.perimeter.circle.model import BaseCirclePerimeter
from bikipy.perimeter.polygon.base import BasePolygonPerimeter
from bikipy.perimeter.polygon.triangle import TrianglePerimeter


class AbstractComputeRayOffsetFilter(AbstractComputePerimeterBooleanIndex, ABC):
    ray_start_points: Np2DArrayFp64
    ray_travel_direction_points: Np2DArrayFp64
    max_radians: float

    heuristic_data_sources = ("ray_start_points", "ray_travel_direction_points", "max_radians")
    heuristic_data_sources_all_required = True

    @computed_field  # type: ignore[misc]
    @property
    def perimeter_to_boolean_index(self) -> dict[BasePerimeter, Np1DArrayBool]:
        return {self.perimeter: self.result}
