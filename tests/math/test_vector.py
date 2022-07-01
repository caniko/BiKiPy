import numpy as np


def test_unit_vector():
    single_vector = np.array((0, 10))

    np.testing.assert_almost_equal((0, 1), unit_vector(single_vector))
    np.testing.assert_almost_equal(((0, 1),), unit_vector(single_vector, force_1_dim=True))

    multiple_vectors = np.array(((0, 10), (10, 0)))
    unit_of_multiple = np.array(((0, 1), (1, 0)))

    np.testing.assert_almost_equal(unit_of_multiple, unit_vector(multiple_vectors))


def test_orthogonal_vector():
    single_vector = np.array((0, 100))
    orthogonal_of_single = np.array((-1, 0))

    np.testing.assert_almost_equal(orthogonal_unit_vector(single_vector), orthogonal_of_single)

    multiple_vectors = np.array(((0, 10), (10, 0)))
    orthogonal_of_multiple = np.array(((-1, 0), (0, 1)))

    np.testing.assert_almost_equal(orthogonal_unit_vector(multiple_vectors), orthogonal_of_multiple)


def test_dot_axis_1_1d():
    vectors_a = np.array(((1, 1), (2, 2)))
    vectors_b = np.array(((3, 3), (1, 1)))

    np.testing.assert_almost_equal((6, 4), dot_axis_1_1d(vectors_a, vectors_b))


def test_intersection_between_two_lines():
    np.testing.assert_almost_equal(intersection_between_two_lines((1, 0), (0, 1), (-1, 0), (0, -1)), (1, 1))
