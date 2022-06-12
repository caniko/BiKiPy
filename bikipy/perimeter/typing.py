from bikipy.perimeter import ParallelogramPerimeter
from bikipy.perimeter.base import PerimeterSet
from bikipy.perimeter.polygon.base import PolygonPerimeter
from bikipy.perimeter.polygon.triangular import TriangularPerimeter
from bikipy.perimeter.radial.circle import CirclePerimeter

AnyPerimeter = PerimeterSet | CirclePerimeter | PolygonPerimeter | ParallelogramPerimeter | TriangularPerimeter
