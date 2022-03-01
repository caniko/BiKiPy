from bikipy.math.geometry import expand_parallelogram


def test_expand_parallelogram():
    result = expand_parallelogram(
        ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)), offset=1, y_inverted=False
    )
    expected = ((-1.0, 2.0), (2.0, 2.0), (2.0, -1.0), (-1.0, -1.0))
    assert all(coordinate in expected for coordinate in result)
