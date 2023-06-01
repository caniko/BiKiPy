from functools import cached_property

from matplotlib.axes import Axes
from pydantic_numpy.dtype import NDArrayFp64

from bikipy import runtime_settings
from bikipy.core.video import VideoMetadata
from bikipy.feature.compute import AbstractComputeBooleanIndex
from bikipy.feature.tolerance.single import single_node_tolerance_model
from bikipy.perimeter.base import SinglePerimeter
from bikipy.utils.math.vector import unit_vector
from bikipy.utils.plot import BOTTOM_LEGEND_KWARGS
from tests.test_data.perimeter.objects import rectangle_perimeter_rectangle_test_object


class ComputeInLineOfSight(AbstractComputeBooleanIndex):
    perimeter: SinglePerimeter = ...
    ray_start_point: NDArrayFp64 = ...
    ray_travel_direction_point: NDArrayFp64 = ...
    max_radians: float = ...


def test_compute_compute_in_line_of_sight():
    perimeter = rectangle_perimeter_rectangle_test_object
    c_ilos = ComputeInLineOfSight(perimeter=perimeter, ray_start_point=perimeter.)
