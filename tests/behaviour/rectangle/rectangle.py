import numpy as np
from numpy.testing import assert_allclose

from bikipy.behaviour.rectangle.rectangle import _compute_quadrant_grid_coordinates


def test_compute_quadrant_grid_coordinates():
    assert_allclose(
        [
            [0, 0],
            [20, 0],
            [20, 20],
            [0, 20]
        ],
        _compute_quadrant_grid_coordinates((2, 2), np.array([40, 40]))[(1, 1)]
    )
