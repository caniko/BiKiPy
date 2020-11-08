from typing import Sequence
import numpy as np


# TODO: Sanity checks, is the array 2D, etc


def unit_vector(row_vectors: Sequence, force_1_dim: bool = False) -> np.ndarray:
    """
    Computes unit vector, i.e. vector/<norm of the vector>

    Parameters
    ----------
    row_vectors: np.ndarray-like
        Array of row vector(s)

    force_1_dim: bool
        If True, make sure that the results are sent back as a Sequence within an array
        important when working with single vectors and functions that expect
        Sequence of vectors

    Returns
    -------
    np.ndarray
    """
    row_vectors = np.asanyarray(row_vectors)

    if len(row_vectors.shape) != 2:
        # Single vector
        result = row_vectors / np.linalg.norm(row_vectors)

        if force_1_dim:
            return np.expand_dims(result, axis=0)

        return result

    # Multiple vectors
    return (row_vectors.T / np.linalg.norm(row_vectors, axis=1)).T


def orthogonal_vector(vector: Sequence) -> np.ndarray:
    """
    Computes the orthogonal unit vector of the given 2D vector

    Parameters
    ----------
    vector: np.ndarray-like
        Array of row vector(s)

    Returns
    -------
    np.ndarray
    """
    vector = np.asanyarray(vector)

    if vector.shape == (2,):
        return unit_vector((-vector[1], vector[0]))
    else:
        return unit_vector(np.array((-vector.T[1], vector.T[0])).T)


def dot_prod_along_axis_1(vector_a, vector_b):
    # np.einsum("ij,ij->i", vector_a, vector_b)
    return np.nansum(vector_a * vector_b, axis=1)


def normal_from_line_to_point(
    line_vector: Sequence, line_start: Sequence, point: Sequence
):
    """
    Computes the magnitude of two vectors. Vector nr. 1 with line vector as unit,
    from 'line_start' to the beginning of the normal, and,
    vector nr. 2 with orthogonal to line vector as unit,
    from beginning of normal to the given 'point'.

    Parameters
    ----------
    line_vector: np.ndarray-like
    line_start: np.ndarray-like
    point: np.ndarray-like

    Returns
    -------

    """
    line_unit_vector = unit_vector(line_vector)
    orthogonal_unit_line_vector = orthogonal_vector(line_unit_vector)

    lhs = np.array(
        (
            (line_unit_vector[0], -orthogonal_unit_line_vector[0]),
            (line_unit_vector[1], -orthogonal_unit_line_vector[1]),
        )
    )
    rhs = np.array(
        (
            -point[0] + line_start[0],
            -point[1] + line_start[1],
        )
    )
    sol = np.linalg.solve(lhs, rhs)

    return sol


def distance_between_line_and_point(*args, **kwargs):
    """
    Compute distance between point and a line.

    Wrapper around normal_from_line_to_point()
    """
    return normal_from_line_to_point(*args, **kwargs)[1]


def closest_line_to_point(
    line_vectors: Sequence, line_starts: Sequence, point: Sequence
):
    distances = [
        distance_between_line_and_point(line_vector, line_start, point)
        for line_vector, line_start in zip(line_vectors, line_starts)
    ]
    closest_index = np.where(np.argsort(distances) == 0)[0][0]
    return distances[closest_index], closest_index


def intersection_between_two_lines(
    vector_a: Sequence,
    vector_b: Sequence,
    vector_a_start: Sequence,
    vector_b_start: Sequence,
) -> np.ndarray:
    """

    Parameters
    ----------
    vector_a
    vector_b
    vector_a_start
    vector_b_start

    Returns
    -------

    """
    rhs = np.array(((vector_a[0], -vector_b[0]), (vector_a[1], -vector_b[1])))
    lhs = np.array(
        (
            (vector_b_start[0] - vector_a_start[0],),
            (vector_b_start[1] - vector_a_start[1],),
        )
    )
    return np.linalg.solve(rhs, lhs).T[0]
