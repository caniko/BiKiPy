from typing import Callable

import numba
import numpy as np
from matplotlib import pyplot as plt
from numba import jit, njit
from numpy.linalg import LinAlgError
from pydantic import validate_arguments
from pydantic_numpy import NDArray

from bikipy import ENABLE_NUMBA
from bikipy.core.typing import NDArrayFp64
from bikipy.utils.collection_utils import evenly_spaced_indices


@validate_arguments
def unit_vector(row_vectors: NDArrayFp64, force_1_dim: bool = False) -> NDArrayFp64:
    """
    Computes unit vector, i.e. vector/<norm of the vector>

    Parameters
    ----------
    row_vectors: NDArrayFp64-like
        Array of row vector(s)

    force_1_dim: bool
        If True, make sure that the results are sent back as a NDArrayFp64 within an array
        important when working with single vectors within functions that expect
        a NDArrayFp64 of vectors

    Returns
    -------
    All unit vectors along the rows of row_vectors
    """
    if len(row_vectors.shape) != 2:
        # Single vector
        result = row_vectors / np.linalg.norm(row_vectors)

        if force_1_dim:
            return np.expand_dims(result, axis=0)

        return result

    # Multiple vectors
    return (row_vectors.T / np.linalg.norm(row_vectors, axis=1)).T


def orthogonal_unit_vector(vector: NDArrayFp64) -> NDArrayFp64:
    """
    Computes the orthogonal unit vector of the given 2D vector

    Parameters
    ----------
    vector: NDArrayFp64-like
        Array of row vector(s)

    Returns
    -------
    NDArrayFp64
    """
    if vector.shape == (2,):
        return unit_vector((-vector[1], vector[0]))
    else:
        return unit_vector(np.array((-vector.T[1], vector.T[0])).T)


def dot_axis_1_1d(vector_a: NDArrayFp64, vector_b: NDArrayFp64) -> NDArrayFp64:
    # np.einsum("ij,ij->i", vector_a, vector_b)
    return np.nansum(vector_a * vector_b, axis=1)


def normal_from_line_to_point(line_vector: NDArrayFp64, line_start: NDArrayFp64, point: NDArrayFp64):
    """
    Computes the magnitude of two vectors. Vector nr. 1 with line vector as unit,
    from 'line_start' to the beginning of the normal, and,
    vector nr. 2 with orthogonal to line vector as unit,
    from beginning of normal to the given 'point'.

    Parameters
    ----------
    line_vector: NDArrayFp64-like
    line_start: NDArrayFp64-like
    point: NDArrayFp64-like

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


def distance_between_line_and_point(*args, **kwargs) -> NDArrayFp64:
    """
    Compute distance between point and a line.

    Wrapper around normal_from_line_to_point()
    """
    return normal_from_line_to_point(*args, **kwargs)[1]


@validate_arguments
def nearest_point_on_line_segment_to_coordinates(
    line_segment_start: NDArrayFp64, line_segment_end: NDArrayFp64, coordinates: NDArrayFp64, inspect: bool = False
) -> NDArrayFp64:
    # https://stackoverflow.com/a/47484153/9793651
    start_end_vector = line_segment_end - line_segment_start
    start_coordinates_vectors = coordinates - line_segment_start

    interpolation_param = dot_axis_1_1d(start_end_vector, start_coordinates_vectors) / np.linalg.norm(start_end_vector)

    filtered_ip = np.where(interpolation_param < 0, 0, interpolation_param)  # lowest value is 0
    filtered_ip = np.where(filtered_ip > 1, 1, filtered_ip)  # highest values is 1

    result = line_segment_start + (filtered_ip * start_end_vector[:, None]).T

    if inspect:
        fig, axes = plt.subplots(3, 3)
        axes = np.array(axes)

        for i, ax in zip(evenly_spaced_indices(coordinates, 9), axes.reshape(-1)):
            ax.plot(*np.vstack((line_segment_start, line_segment_end)).T)
            ax.scatter(*result[i])
            ax.scatter(*coordinates[i])

        plt.tight_layout()
        plt.show()

    return result


@validate_arguments
def closest_line_to_point(line_vectors: NDArrayFp64, line_starts: NDArrayFp64, point: NDArrayFp64):
    distances = [
        distance_between_line_and_point(line_vector, line_start, point)
        for line_vector, line_start in zip(line_vectors, line_starts)
    ]
    closest_index = np.where(np.argsort(distances) == 0)[0][0]
    return distances[closest_index], closest_index


@validate_arguments
def intersection_between_two_lines(
    vector_a: NDArrayFp64,
    vector_b: NDArrayFp64,
    vector_a_start: NDArrayFp64,
    vector_b_start: NDArrayFp64,
) -> NDArrayFp64 | None:
    """
    Compute the intersection between two lines designated by a starting point
    and a direction/unit vector

    returns False if no intersection

    FIXME: Does not work when the intersection is on (0, 0); very rare case

    Parameters
    ----------
    vector_a: NDArrayFp64
        Unit vector of line A
    vector_b: NDArrayFp64
        Unit vector of line B
    vector_a_start: NDArrayFp64
        Origin or starting point of line A
    vector_b_start: NDArrayFp64
        Origin or starting point of line B

    Returns
    -------
    NDArrayFp64: The intersection point between lina A and B
    """
    rhs = np.array(((vector_a[0], -vector_b[0]), (vector_a[1], -vector_b[1])))
    lhs = np.array(
        (
            (vector_b_start[0] - vector_a_start[0],),
            (vector_b_start[1] - vector_a_start[1],),
        )
    )
    try:
        return np.linalg.solve(rhs, lhs).T[0]
    except LinAlgError:
        return None


@validate_arguments
def numpy_bin(data: NDArray, axis: int, bin_step: int, bin_size: int, reducer: Callable = np.nanmean) -> NDArray:
    arg_dims = np.arange(data.ndim)
    arg_dims[0], arg_dims[axis] = arg_dims[axis], arg_dims[0]
    data = data.transpose(arg_dims)
    data = [
        reducer(np.take(data, np.arange(int(i * bin_step), int(i * bin_step + bin_size)), 0), 0)
        for i in np.arange(data.shape[axis] // bin_step)
    ]
    return np.array(data).transpose(arg_dims)


def rotation_matrix_from_radians(radians: NDArrayFp64) -> NDArrayFp64:
    cos, sin = np.cos(radians), np.sin(radians)
    return np.ascontiguousarray(([cos, -sin], [sin, cos])).transpose(2, 0, 1)


if ENABLE_NUMBA:

    # rotation_matrix_from_radians = jit(cache=True)(rotation_matrix_from_radians)
    # dot_axis_1_1d = njit(cache=True)(dot_axis_1_1d)   https://github.com/numba/numba/issues/1269
    orthogonal_unit_vector = njit(cache=True)(orthogonal_unit_vector)

    def rotate_vectors_with_angle(vectors: NDArrayFp64, angles: NDArrayFp64) -> NDArrayFp64:
        return rotate_vectors_with_angle(vectors, rotation_matrix_from_radians(angles))

    @njit(parallel=True, nogil=True, cache=True)
    def rotate_vectors_with_rotation_matrix(vectors: NDArrayFp64, rotation_matrices: NDArrayFp64) -> NDArrayFp64:
        result = np.empty_like(vectors)
        for i in numba.prange(len(vectors)):
            result[i] = np.dot(vectors[i], rotation_matrices[i])
        return result

else:

    def rotate_vectors_with_angle(vectors: NDArrayFp64, angles: NDArrayFp64) -> NDArrayFp64:
        return np.array(
            [
                np.dot(vector, rotation_matrix)
                for vector, rotation_matrix in zip(vectors, rotation_matrix_from_radians(angles))
            ]
        )
