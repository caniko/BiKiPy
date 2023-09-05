import numpy as np

from bikipy.math.confinement.polygon import parallel_point_inside_polygon


def test_parallel_point_inside_polygon():
    inside, outside, on_border = parallel_point_inside_polygon(
        ab_mid_corner=np.array((0, 0)),
        corner_a=np.array((0, 1)),
        corner_b=np.array((1, 0)),
        coordinates=np.array(((0.5, 0.5), (1.5, 1.5), (1.0, 1.0))),
    )

    assert inside
    assert not outside

    assert on_border
