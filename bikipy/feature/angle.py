from collections.abc import Sequence
from typing import Union
from warnings import warn

import numpy as np
from numba import njit

from bikipy.utils.math import dot_prod_along_axis_1, fast_unit_vector, unit_vector

POINT_NAME_TO_INDEX = {"a": 0, "b": 1, "c": 2}


def _find_median_vector(row_vectors: np.ndarray) -> np.ndarray:
    """
    Computes the median point from a row vectors

    Median of each component -> combine medians to create median point. Note that this point doesn't exist

    Parameters
    ----------
    row_vectors: np.ndarray
        Array of row vectors

    Returns
    -------
    np.ndarray
    """
    return np.array([np.median(component) for component in row_vectors.T])


def clockwise_angel_2d(
    start_vector: Sequence,
    end_vector: Sequence,
) -> np.ndarray:
    """
    Computes the counterclockwise angle, [0, 2pi], from start to end in radians

    :param start_vector: Array of row vectors in which "the clock starts turning", counterclockwise
    :param end_vector: Array of row vectors in which the clock stops
    :type start_vector: np.ndarray
    :type end_vector: np.ndarray
    :return: counterclockwise angle between start and end vector per frame
    :rtype: np.ndarray

    >>> clockwise_angel_2d((1, 0), (0, 1))
    1.5707963267948966      # pi / 2.
    >>> clockwise_angel_2d((1, 0), (1, 0))
    0.0
    >>> clockwise_angel_2d((1, 0), (-1, 0))
    3.141592653589793       # pi"""

    start_vector = unit_vector(start_vector, force_1_dim=True)
    end_vector = unit_vector(end_vector, force_1_dim=True)

    length_start = len(start_vector)
    length_end = len(end_vector)
    if length_end != length_start and not (length_end == 1 or length_start == 1):
        msg = (
            f"start and vector can either be constant, or have the same length. "
            f"start = {length_start}, end = {length_end}"
        )
        raise ValueError(msg)

    # Compute determinants and store them in a vertical stack
    determinants = np.array(
        [
            np.linalg.det(
                np.vstack(
                    (
                        (end_vector if length_end == 1 else end_vector[i]),
                        (start_vector if length_start == 1 else start_vector[i]),
                    )
                )
            )
            for i in range(max(length_start, length_end))
        ]
    )

    dot_products = dot_prod_along_axis_1(end_vector, start_vector)
    angles = np.arctan2(np.abs(determinants), dot_products)
    angles[determinants < 0.0] = 2.0 * np.pi - angles[determinants < 0.0]
    return angles


def alternative_inner_angle(a_vector: Sequence, b_vector: Sequence) -> np.ndarray:
    """
    Computes the inner angle between two vectors, a and b, in radians

    .. math::
        \theta = \cos^{-1} \Big( \frac{\mathbf{a} \cdot \mathbf{b}}{|\mathbf{a}||\mathbf{b}|} \Big)

    :param a_vector: Array of row vectors in which "the clock starts turning" counter counterclockwise
    :param b_vector: Array of row vectors in which the clock stops
    :type a_vector: np.ndarray
    :type b_vector: np.ndarray
    :return: Inner angle between a and b vector per frame
    :rtype: np.ndarray
    """
    a_unit_vector = unit_vector(a_vector, force_1_dim=True)
    b_unit_vector = unit_vector(b_vector, force_1_dim=True)

    return np.arccos(
        dot_prod_along_axis_1(a_unit_vector, b_unit_vector)
        / (np.linalg.norm(a_unit_vector, axis=1) * np.linalg.norm(b_unit_vector, axis=1))
    )


@njit(cache=True, nogil=True)
def inner_angle(vector_set_1, vector_set_2):
    """Returns the angle in radians between given vectors"""
    result = []
    for i in range(len(vector_set_2)):
        v1_u = fast_unit_vector(vector_set_1[i])
        v2_u = fast_unit_vector(vector_set_2[i])
        minor = np.linalg.det(np.stack((v1_u[-2:], v2_u[-2:])))
        if minor == 0:
            sign = 1
        else:
            sign = -np.sign(minor)
        dot_p = np.dot(v1_u, v2_u)
        dot_p = min(max(dot_p, -1.0), 1.0)
        result.append(sign * np.arccos(dot_p))
    return np.array(result)


def compute_angles_from_vectors(
    row_vectors_point_a: np.ndarray,
    row_vectors_point_b: np.ndarray,
    row_vectors_point_c: np.ndarray,
    median_points: Union[str, Sequence, None] = None,
    method: str = "inner",
    degrees: bool = False,
) -> np.ndarray:
    """
    Computes the angle between three groups of vectors

    :param row_vectors_point_a: Array of row vectors
    :param row_vectors_point_b: Array of row vectors that is the joint between the two other groups of vectors
    :param row_vectors_point_c: Array of row vectors
    :param median_points: Anchor one or several points to their respective median. Information about median computation in _find_median_vector()
    :param method: The method for computing angle, supported methods are inner; counterclockwise.
    :param degrees: If True, convert resulting angle data to degrees
    :type row_vectors_point_a: np.ndarray
    :type row_vectors_point_b: np.ndarray
    :type row_vectors_point_c: np.ndarray
    :type median_points: Iterable, str
    :type method: str
    :type degrees: bool
    :return: Angle per frame
    :rtype: np.ndarray
    """

    points = [
        np.asarray(row_vectors_point_a),
        np.asarray(row_vectors_point_b),
        np.asarray(row_vectors_point_c),
    ]
    if any(point.dtype == "object" for point in points):
        warn("At least one of the arrays consists solely of NaN (Not a Number) objects")
        return np.full((points[0].size,), np.nan)

    if isinstance(median_points, str):
        index = POINT_NAME_TO_INDEX[median_points]
        points[index] = _find_median_vector(points[index])
    elif median_points:
        try:
            for label in median_points:
                index = POINT_NAME_TO_INDEX[label]
                points[index] = _find_median_vector(points[index])
        except KeyError as e:
            msg = "When median_points is an Iterable it must store either a, b and/or c"
            raise KeyError(msg) from e
        except TypeError as e:
            msg = "median_points has to be list, string or None"
            raise TypeError(msg) from e

    try:
        computation = ANGLE_METHOD_TO_FUNC[method.lower()](
            points[1] - points[0], points[2] - points[1]  # AB Vector  # BC Vector
        )
    except KeyError as e:
        msg = f"{method} is not a supported method. Supported methods are {ANGLE_METHOD_TO_FUNC.keys()}"
        raise ValueError(msg) from e

    if degrees:
        return np.rad2deg(computation)

    return computation


def angles_between_0_2pi(angles: np.array):
    angles = np.asarray(angles)

    boolean_indexes = np.abs(angles) >= 2.0 * np.pi
    angles[boolean_indexes] = 2.0 * np.pi - angles[boolean_indexes]

    return angles


ANGLE_METHOD_TO_FUNC = {
    "inner": inner_angle,
    "counterclockwise": clockwise_angel_2d,
}
