import numpy as np

from bikipy.feature.attention.proximity import proximity_filter
from bikipy.utils.math.geometry import expand_rectangle
from tests.test_data.perimeter.parallelogram.get_perimeter import (
    rectangle_perimeter_coco_test_object,
)


def test_proximity_filter():
    coordinates_inside_perimeter = expand_rectangle(rectangle_perimeter_coco_test_object.vertices_in_meters, -1.0)
    coordinates_outside_perimeter = expand_rectangle(rectangle_perimeter_coco_test_object.vertices_in_meters, 1.0)

    perimeter_border_normal_pixels = 50
    border_vertices = rectangle_perimeter_coco_test_object.expand(perimeter_border_normal_pixels).vertices_in_meters

    coordinates_inside_border = expand_rectangle(border_vertices, -1.0)
    coordinates_outside_border = expand_rectangle(border_vertices, 1.0)

    assert np.all(
        proximity_filter(
            perimeter=rectangle_perimeter_coco_test_object,
            inside_perimeter_border=coordinates_inside_border,
            outside_perimeter=coordinates_outside_perimeter,
            perimeter_border_normal_pixels=perimeter_border_normal_pixels,
        )[0]
    ), "Coordinates should be in proximity"

    assert not np.all(
        proximity_filter(
            perimeter=rectangle_perimeter_coco_test_object,
            inside_perimeter_border=coordinates_outside_border,
            outside_perimeter=coordinates_inside_perimeter,
            perimeter_border_normal_pixels=perimeter_border_normal_pixels,
        )[0]
    ), "Coordinates should not be in proximity"
