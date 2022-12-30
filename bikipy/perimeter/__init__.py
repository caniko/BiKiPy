from bikipy.perimeter.polygon import PolygonPerimeter, RectanglePerimeter
from bikipy.perimeter.circle import CirclePerimeter


PERIMETERS = (PolygonPerimeter, RectanglePerimeter, CirclePerimeter)
PERIMETER_CLASS_NAME_TO_CLASS = {p.__name__: p for p in PERIMETERS}
