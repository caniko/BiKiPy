from bikipy.perimeter.circle import CirclePerimeter
from bikipy.perimeter.polygon.base import PolygonPerimeter
from bikipy.perimeter.polygon.rectangle import RectanglePerimeter

PERIMETERS = (PolygonPerimeter, RectanglePerimeter, CirclePerimeter)
PERIMETER_CLASS_NAME_TO_CLASS = {p.__name__: p for p in PERIMETERS}
