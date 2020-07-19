import numpy as np

from bikipy.compute.midpoints import recursive_midpoint, compute_midpoint


point_1 = [(2, 0), (3, 0), (0, 4)]
point_2 = [(4, 0), (7, 0), (0, 10)]
point_3 = [(5, 0), (9, 0), (0, 9)]


def test_recursive_midpoint():
    result = recursive_midpoint((point_1, point_2, point_3))
    expected = np.array([(4, 0), (7, 0), (0, 8)])

    assert np.all(np.isclose(result, expected)), result


def test_compute_midpoint():
    result = compute_midpoint(point_1, point_2)
    expected = np.array([(3, 0), (5, 0), (0, 7)])

    assert np.all(np.isclose(result, expected)), result
