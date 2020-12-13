import numpy as np

from bikipy.feature.midpoint import compute_midpoint, recursive_midpoint

point_1 = np.array(((2, 0), (3, 0), (0, 4)))
point_2 = np.array(((4, 0), (7, 0), (0, 10)))
point_3 = np.array(((5, 0), (9, 0), (0, 9)))


def test_recursive_midpoint():
    result = recursive_midpoint((point_1, point_2, point_3))
    expected = np.array(((4, 0), (7, 0), (0, 8)))

    np.testing.assert_allclose(result, expected), result


def test_compute_midpoint():
    result = compute_midpoint(point_1, point_2)
    expected = np.array(((3, 0), (5, 0), (0, 7)))

    np.testing.assert_allclose(result, expected), result
