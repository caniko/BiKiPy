from bikipy.math.point_in_polygon import points_in_parallelogram


def test_points_in_parallelogram():
    # FIXME: Does not work when the intersection is on (0, 0); very rare case
    assert points_in_parallelogram(
        corner_point=(0, 0), point_a=(0, 1), point_b=(1, 0), coordinates=((0.5, 0.5),)
    )
