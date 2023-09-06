import numpy as np
from numpy.testing import assert_allclose


def test_compute_quadrant_grid_coordinates():
    assert_allclose(
        [[0, 0], [20, 0], [20, 20], [0, 20]], _compute_quadrant_grid_coordinates((2, 2), np.array([40, 40]))[(1, 1)]
    )
    assert_allclose(
        [[5, 0], [25, 0], [25, 20], [5, 20]],
        _compute_quadrant_grid_coordinates((2, 2), np.array([40, 40]), translation=np.array([5, 0]))[(1, 1)],
    )
