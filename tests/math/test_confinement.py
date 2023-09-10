import numpy as np

from bikipy.math.confinement.polygon import parallel_point_inside_polygon


def test_parallel_point_inside_polygon():
    inside, almost_border, on_border, outside = parallel_point_inside_polygon(
        points=np.array(((1.0, 1.0), (0.99, 0.99), (2.0, 2.0), (2.5, 2.5))),
        polygon=np.array(((0.0, 0.0), (2.0, 2.0))),
    )

    assert inside
    assert almost_border
    assert on_border
    assert not outside
