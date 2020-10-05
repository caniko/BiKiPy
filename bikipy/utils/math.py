from typing import Sequence
import numpy as np


def unit_vector(row_vectors: Sequence) -> np.ndarray:
    """
    Computes unit vector, i.e. vector/<norm of the vector>

    Parameters
    ----------
    row_vectors: np.ndarray-like
        Array of row vector(s)

    Returns
    -------
    np.ndarray
    """
    row_vectors = np.asanyarray(row_vectors)
    return row_vectors / np.linalg.norm(row_vectors)


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
        return np.apply_along_axis(lambda x: unit_vector((-x[1], x[0])), 1, vector)


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
    unit_line_vector = unit_vector(line_vector)
    orthogonal_unit_line_vector = orthogonal_vector(unit_line_vector)

    lhs = np.array(
        (
            (unit_line_vector[0], -orthogonal_unit_line_vector[0]),
            (unit_line_vector[1], -orthogonal_unit_line_vector[1]),
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


def find_intersection_between_two_vectors(
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
