import numpy as np

from bikipy.feature.attention import proximity_filter

from tests.test_data.perimeter.get_perimeter import perimeter_object


def test_proximity_filter():
    proximity_filter(
        perimeter=perimeter_object,
        inside_perimeter_border=((3, 3), (5, 4)),
        outside_perimeter=((2, 2), (3, 3)),
        perimeter_border_normal_pixel_magnitude=2,
        inspect=True
    )
