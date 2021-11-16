import numpy as np

from bikipy.math.geometry import expand_parallelogram


def test_expand_parallelogram():
    result = expand_parallelogram(
        ((0, 0), (1, 0), (1, 1), (0, 1)), offset=1, y_inverted=False, as_array=True
    )
    assert np.all(
        result
        == np.array(
            ((-1.0, 2.0), (2.0, 2.0), (2.0, -1.0), (-1.0, -1.0)),
        )
    )
