from bikipy.perimeter import CircleVariableRadiusPerimeter
from bikipy.perimeter.base import PerimeterCLS

_polygon_shapes: set[str] = {"rectangle", "polygon"}

MAKESENSE_SHAPES = ("circle", "polygon", "rectangle")

PERIMETER_CLASS_REQUIRE_INTERFACE_SETTINGS: set[PerimeterCLS] = {CircleVariableRadiusPerimeter}
