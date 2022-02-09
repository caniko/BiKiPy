from typing import Union

from bikipy.perimeter.base import PerimeterSet
from bikipy.perimeter.polygon.base import PolygonPerimeter

Perimeter2D = Union[PolygonPerimeter, PerimeterSet]
