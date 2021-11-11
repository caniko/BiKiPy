import numpy as np

from bikipy.feature.attention import proximity_filter
from bikipy.math.geometry import expand_parallelogram
from bikipy.utils.misc import to_tuple

from tests.test_data.perimeter.parallelogram.get_perimeter import perimeter_object


def test_proximity_filter():
    coordinates_inside_perimeter = expand_parallelogram(to_tuple(perimeter_object.corners), -1.0)
    coordinates_outside_perimeter = expand_parallelogram(to_tuple(perimeter_object.corners), 1.0)

    perimeter_border_normal_pixel_magnitude = 50
    border_corners = perimeter_object.border(
        perimeter_border_normal_pixel_magnitude
    ).corners

    coordinates_inside_border = expand_parallelogram(to_tuple(border_corners), -1.0)
    coordinates_outside_border = expand_parallelogram(to_tuple(border_corners), 1.0)

    assert np.all(
        proximity_filter(
            perimeter=perimeter_object,
            inside_perimeter_border=coordinates_inside_border,
            outside_perimeter=coordinates_outside_perimeter,
            perimeter_border_normal_pixel_magnitude=perimeter_border_normal_pixel_magnitude,
        )[0]
    ), "Coordinates should be in proximity"

    assert not np.all(
        proximity_filter(
            perimeter=perimeter_object,
            inside_perimeter_border=coordinates_outside_border,
            outside_perimeter=coordinates_inside_perimeter,
            perimeter_border_normal_pixel_magnitude=perimeter_border_normal_pixel_magnitude,
        )[0]
    ), "Coordinates should not be in proximity"
