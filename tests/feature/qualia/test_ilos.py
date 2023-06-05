import numpy as np

from bikipy.feature.qualia.axioms.ilos import ComputeInLineOfSight
from tests.test_data.perimeter.test_perimeter_readers import (
    rectangle_perimeter_rectangle_test_object,
)


def test_compute_compute_in_line_of_sight():
    perimeter = rectangle_perimeter_rectangle_test_object["Center"]
    # rightmost_midpoint
    mid_x, mid_y = perimeter.pixel_graph.vertex_midpoints[0]
    corner_x, corner_y = perimeter.vertices_in_pixels[0]

    # Ray that hits rectangle in the middle of an edge
    mid_ray_start_point = [mid_x + 50, mid_y]
    mid_ray_direction_point = [mid_x + 40, mid_y]

    # Ray that starts at a point offset from the middle of an edge, while the ray is perpendicular
    perpendicular_direction_point = [mid_x + 50, mid_y + 50]

    # Ray that starts 1 pixel offset from a corner
    corner_start_point = [corner_x + 50, corner_y]
    corner_direction_point = [corner_x + 40, corner_y]
    corner_start_point_off_by_1 = [corner_x + 50, corner_y + 1]
    corner_direction_point_off_by_1 = [corner_x + 40, corner_y + 1]

    c_ilos = ComputeInLineOfSight(
        perimeter=perimeter,
        ray_start_point=np.array(
            [mid_ray_start_point, mid_ray_start_point, corner_start_point, corner_start_point_off_by_1]
        ),
        ray_travel_direction_point=np.array(
            [
                mid_ray_direction_point,
                perpendicular_direction_point,
                corner_direction_point,
                corner_direction_point_off_by_1,
            ]
        ),
        label="test",
        # max_radians=np.pi / 4,
        max_radians=0,
        tolerance_modelling=False,
    )

    # fig, ax = plt.subplots()
    # c_ilos.plot(ax)
    # plt.show()

    assert c_ilos.result[0]
    assert not c_ilos.result[1]

    assert c_ilos.result[2]
    assert not c_ilos.result[3]
