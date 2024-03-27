from bikipy.perimeter.base import BaseSinglePerimeter
from bikipy.perimeter.circle.model import (
    CircleFixedRadiusPerimeter,
    CircleVariableRadiusPerimeter,
)
from bikipy.perimeter.polygon.rectangle import RectanglePerimeter

PERIMETERS = (
    BaseSinglePerimeter,
    RectanglePerimeter,
    CircleVariableRadiusPerimeter,
    CircleFixedRadiusPerimeter,
)
PERIMETER_CLASS_NAME_TO_CLASS = {p.__name__: p for p in PERIMETERS}
