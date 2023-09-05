import numpy as np

from bikipy.math import points_in_rectangle


def test_points_in_rectangle():
    # FIXME: Does not work when the intersection is on (0, 0); very rare case
    inside, outside, on_border = points_in_rectangle(
        ab_mid_corner=np.array((0, 0)),
        corner_a=np.array((0, 1)),
        corner_b=np.array((1, 0)),
        coordinates=np.array(((0.5, 0.5), (1.5, 1.5), (1.0, 1.0))),
    )

    assert inside
    assert not outside

    assert on_border
