from bikipy.perimeter.base import BaseSinglePerimeter
from bikipy.perimeter.circle import (
    CircleFixedRadiusPerimeter,
    CircleVariableRadiusPerimeter,
)
from bikipy.perimeter.polygon.base import PolygonPerimeter
from bikipy.perimeter.polygon.rectangle import RectanglePerimeter

PERIMETERS = (
    BaseSinglePerimeter,
    PolygonPerimeter,
    RectanglePerimeter,
    CircleVariableRadiusPerimeter,
    CircleFixedRadiusPerimeter,
)
PERIMETER_CLASS_NAME_TO_CLASS = {p.__name__: p for p in PERIMETERS}
