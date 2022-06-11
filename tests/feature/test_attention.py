import numpy as np

from bikipy.feature.attention import proximity_filter
from bikipy.utils.math.geometry import expand_parallelogram
from tests.test_data.perimeter.parallelogram.get_perimeter import (
    parallelogram_perimeter_coco_test_object,
)


def test_proximity_filter():
    coordinates_inside_perimeter = expand_parallelogram(parallelogram_perimeter_coco_test_object.corners, -1.0)
    coordinates_outside_perimeter = expand_parallelogram(parallelogram_perimeter_coco_test_object.corners, 1.0)

    perimeter_border_normal_pixel_magnitude = 50
    border_corners = parallelogram_perimeter_coco_test_object.expand(perimeter_border_normal_pixel_magnitude).corners

    coordinates_inside_border = expand_parallelogram(border_corners, -1.0)
    coordinates_outside_border = expand_parallelogram(border_corners, 1.0)

    assert np.all(
        proximity_filter(
            perimeter=parallelogram_perimeter_coco_test_object,
            inside_perimeter_border=coordinates_inside_border,
            outside_perimeter=coordinates_outside_perimeter,
            perimeter_border_normal_pixel_magnitude=perimeter_border_normal_pixel_magnitude,
        )[0]
    ), "Coordinates should be in proximity"

    assert not np.all(
        proximity_filter(
            perimeter=parallelogram_perimeter_coco_test_object,
            inside_perimeter_border=coordinates_outside_border,
            outside_perimeter=coordinates_inside_perimeter,
            perimeter_border_normal_pixel_magnitude=perimeter_border_normal_pixel_magnitude,
        )[0]
    ), "Coordinates should not be in proximity"
