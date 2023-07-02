from bikipy.utils.math.geometry import expand_rectangle


def test_expand_rectangle():
    result = expand_rectangle(
        ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)), x_offset=1, y_offset=1, y_inverted=False
    )
    expected = ((-1.0, 2.0), (2.0, 2.0), (2.0, -1.0), (-1.0, -1.0))
    assert all(coordinate in expected for coordinate in result)
