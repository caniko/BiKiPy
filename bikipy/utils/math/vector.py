from typing import Callable

import numba
import numpy as np
from matplotlib import pyplot as plt
from numba import njit
from numpy.linalg import LinAlgError
from pydantic import validate_arguments
from pydantic_numpy import NDArray
from pydantic_numpy.dtype import NDArrayFp64

from bikipy import runtime_settings
from bikipy.utils.collection_utils import (
    evenly_spaced_indices,
    evenly_spaced_indices_from_sequence,
    flatten_sequence,
)


@validate_arguments
def unit_vector(row_vectors: NDArrayFp64, force_1_dim: bool = False) -> NDArrayFp64:
    """
    Computes unit vector, i.e. vector/<norm of the vector>

    :param row_vectors: Array of row vector(s)
    :param force_1_dim: If True, make sure that the results are sent back as a NDArrayFp64 within an array
        important when working with single vectors within functions that expect
        a NDArrayFp64 of vectors
    :return: All unit vectors along the rows of row_vectors
    """
    if len(row_vectors.shape) != 2:
        # Single vector
        result = row_vectors / np.linalg.norm(row_vectors)

        if force_1_dim:
            return np.expand_dims(result, axis=0)

        return result

    # Multiple vectors
    return (row_vectors.T / np.linalg.norm(row_vectors, axis=1)).T


def orthogonal_vector(row_vectors: NDArrayFp64) -> NDArrayFp64:
    """
    Computes the orthogonal row_vectors of the given 2D row_vectors

    :param row_vectors: Array of row vector(s)
    :type row_vectors: NDArrayFp64-like
    :return: Orthogonal vectors with respect to row_vectors
    """

    # if row_vectors.shape == (2,):
    #     return np.ascontiguousarray((-row_vectors[1], row_vectors[0]))
    return np.ascontiguousarray((-row_vectors.T[1], row_vectors.T[0])).T


def orthogonal_unit_vector(row_vectors: NDArrayFp64) -> NDArrayFp64:
    """
    Computes the orthogonal unit row_vectors of the given 2D row_vectors

    :param row_vectors: Array of row vector(s)
    :return: Orthogonal unit vectors with respect to row_vectors
    """
    return unit_vector(orthogonal_vector(row_vectors))


def dot_axis_1_1d(row_vectors_a: NDArrayFp64, row_vectors_b: NDArrayFp64) -> NDArrayFp64:
    """
    Convenience function to perform dot product of vectors in stored in arrays of row vectors.
    The two row vector arrays must have the same shape; numpy will raise an error in cases when this is not true

    :param row_vectors_a: Array of row vectors
    :param row_vectors_b: Array of row vectors
    :return: Dot product of the row vectors
    """
    # np.einsum("ij,ij->i", vector_a, vector_b)
    return np.nansum(row_vectors_a * row_vectors_b, axis=1)


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
    line_segment_start: NDArrayFp64,
    line_segment_end: NDArrayFp64,
    coordinates: NDArrayFp64,
    inspect: bool = False,
) -> NDArrayFp64:
    # # https://stackoverflow.com/a/47484153/9793651
    start_end_vector = line_segment_end - line_segment_start
    start_coordinate_vectors = coordinates - line_segment_start

    interpolation_param = (
        dot_axis_1_1d(start_end_vector, start_coordinate_vectors) / np.linalg.norm(start_end_vector) ** 2
    )

    filtered_ip = np.where(interpolation_param < 0.0, 0.0, interpolation_param)  # lowest value is 0
    filtered_ip = np.where(filtered_ip > 1.0, 1.0, filtered_ip)  # highest values is 1

    result = line_segment_start + (filtered_ip * start_end_vector[:, None]).T

    if inspect:
        fig, axes = plt.subplots(3, 3)
        axes = flatten_sequence(axes)

        for i, ax in zip(evenly_spaced_indices_from_sequence(coordinates, 9), axes):
            ax.plot(*np.vstack((line_segment_start, line_segment_end)).T)
            ax.scatter(*result[i])
            ax.scatter(*coordinates[i])

        plt.tight_layout()
        plt.show()

    return result


@validate_arguments
def ray_and_line_segment_intersection(
    ray_origins: NDArrayFp64,
    ray_directions: NDArrayFp64,
    line_segment_start: NDArrayFp64,
    line_segment_end: NDArrayFp64,
    return_points: bool = False,
    inspect: bool = False,
    number_of_vectors: int = 150,
) -> NDArrayFp64:
    # Ray-Line Segment Intersection Test in 2D
    # http://bit.ly/1CoxdrG
    v1 = ray_origins - line_segment_start
    v2 = line_segment_end - line_segment_start

    ray_directions = unit_vector(ray_directions)
    v3 = orthogonal_vector(ray_directions)

    t1 = np.cross(v2, v1) / dot_axis_1_1d(v2, v3)
    t2 = dot_axis_1_1d(v1, v3) / dot_axis_1_1d(v2, v3)

    line_segment_intersection_bool = (t1 >= 0.0) & (0.0 <= t2) & (t2 <= 1.0)

    if inspect:
        indices = evenly_spaced_indices(np.sum(line_segment_intersection_bool), number_of_vectors)
        fig, ax = plt.subplots()
        ax.quiver(
            *ray_origins[line_segment_intersection_bool][indices].T,
            *ray_directions[line_segment_intersection_bool][indices].T,
            angles="xy",
            scale_units="xy",
            alpha=runtime_settings.matplotlib_scatter_alpha,
            label="Valid",
            color="b",
        )

        non_intersection_bool = ~line_segment_intersection_bool
        indices = evenly_spaced_indices(np.sum(non_intersection_bool), number_of_vectors)
        ax.quiver(
            *ray_origins[non_intersection_bool][indices].T,
            *ray_directions[non_intersection_bool][indices].T,
            angles="xy",
            scale_units="xy",
            alpha=runtime_settings.matplotlib_scatter_alpha,
            label="Invalid",
            color="r",
        )
        ax.legend()
        ax.plot(*np.vstack([line_segment_start, line_segment_end]).T)
        plt.show()

    if return_points:
        raise NotImplementedError()
        result = np.full_like(ray_origins, np.nan)
        result[line_segment_intersection_bool] = ray_origins + t1 * ray_directions
    else:
        return line_segment_intersection_bool


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
def numpy_bin(
    data: NDArray,
    axis: int,
    bin_step: int,
    bin_size: int,
    reducer: Callable = np.nanmean,
) -> NDArray:
    arg_dims = np.arange(data.ndim)
    arg_dims[0], arg_dims[axis] = arg_dims[axis], arg_dims[0]
    data = data.transpose(arg_dims)
    data = [
        reducer(
            np.take(data, np.arange(int(i * bin_step), int(i * bin_step + bin_size)), 0),
            0,
        )
        for i in np.arange(data.shape[axis] // bin_step)
    ]
    return np.array(data).transpose(arg_dims)


def rotation_matrix_from_radians(radians: NDArrayFp64) -> NDArrayFp64:
    cos, sin = np.cos(radians), np.sin(radians)
    return np.ascontiguousarray(([cos, -sin], [sin, cos])).transpose(2, 0, 1)


def rotate_vectors_with_angle(vectors: NDArrayFp64, angle: NDArrayFp64) -> NDArrayFp64:
    rotation_matrix = rotation_matrix_from_radians(angle)
    return np.array([np.dot(vector, rotation_matrix) for vector in vectors]).transpose(1, 0, 2)


if not runtime_settings.disable_numba:

    # rotation_matrix_from_radians = jit(cache=True)(rotation_matrix_from_radians)
    # dot_axis_1_1d = njit(cache=True)(dot_axis_1_1d)   https://github.com/numba/numba/issues/1269
    orthogonal_unit_vector = njit(cache=True)(orthogonal_unit_vector)

    @njit(parallel=True, nogil=True, cache=True)
    def rotate_vectors_with_rotation_matrix(vectors: NDArrayFp64, rotation_matrices: NDArrayFp64) -> NDArrayFp64:
        result = np.empty_like(vectors)
        for i in numba.prange(len(vectors)):
            result[i] = np.dot(vectors[i], rotation_matrices[i])
        return result
