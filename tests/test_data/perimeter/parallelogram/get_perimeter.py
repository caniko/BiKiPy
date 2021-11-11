from pathlib import Path

from bikipy.math.geometry import expand_parallelogram
from bikipy.perimeter.base import Perimeter
from bikipy.utils.misc import to_tuple

FILE_ROOT = Path(__file__).parent


perimeter_object = Perimeter.from_coco(
    FILE_ROOT / "perimeter.json", image_root=FILE_ROOT, single_obj_return=True
)

coordinates_inside_perimeter = expand_parallelogram(to_tuple(perimeter_object.corners), -1.0, inspect=True)
coordinates_outside_perimeter = expand_parallelogram(to_tuple(perimeter_object.corners), 1.0, inspect=True)

perimeter_border_normal_pixel_magnitude = 50
border_corners = perimeter_object.border(
    perimeter_border_normal_pixel_magnitude
).corners

coordinates_inside_border = expand_parallelogram(to_tuple(border_corners), -1.0, inspect=True)
coordinates_outside_border = expand_parallelogram(to_tuple(border_corners), 1.0, inspect=True)
