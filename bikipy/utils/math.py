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
    Computes the normal vector from point to a line in 2D

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
