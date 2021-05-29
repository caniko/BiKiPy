from collections.abc import Sequence

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
        If True, make sure that the results are sent back as a sequence within an array
        important when working with single vectors within functions that expect
        a sequence of vectors

    Returns
    -------
    All unit vectors along the rows of row_vectors
    """
    row_vectors = np.asarray(row_vectors)

    if len(row_vectors.shape) != 2:
        # Single vector
        result = row_vectors / np.linalg.norm(row_vectors)

        if force_1_dim:
            return np.expand_dims(result, axis=0)

        return result

    # Multiple vectors
    return (row_vectors.T / np.linalg.norm(row_vectors, axis=1)).T


def orthogonal_unit_vector(vector: Sequence) -> np.ndarray:
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
    vector = np.asarray(vector)

    if vector.shape == (2,):
        return unit_vector((-vector[1], vector[0]))
    else:
        return unit_vector(np.array((-vector.T[1], vector.T[0])).T)


def dot_prod_along_axis_1(vector_a: np.ndarray, vector_b: np.ndarray) -> np.ndarray:
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
    orthogonal_unit_line_vector = orthogonal_unit_vector(line_unit_vector)

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


def distance_between_line_and_point(*args, **kwargs) -> np.ndarray:
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
    Compute the intersection between two lines designated by a starting point
    and a direction/unit vector

    FIXME: Does not work when the intersection is on (0, 0); very rare case

    Parameters
    ----------
    vector_a: Sequence
        Unit vector of line A
    vector_b: Sequence
        Unit vector of line B
    vector_a_start: Sequence
        Origin or starting point of line A
    vector_b_start: Sequence
        Origin or starting point of line B

    Returns
    -------
    Sequence: The intersection point between lina A and B
    """
    rhs = np.array(((vector_a[0], -vector_b[0]), (vector_a[1], -vector_b[1])))
    lhs = np.array(
        (
            (vector_b_start[0] - vector_a_start[0],),
            (vector_b_start[1] - vector_a_start[1],),
        )
    )
    return np.linalg.solve(rhs, lhs).T[0]
