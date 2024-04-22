from typing import Optional, Sequence
from warnings import warn

import numpy as np
from numba import njit, prange
from pydantic_numpy.typing import Np1DArrayBool, Np2DArrayFp64

from bikipy import runtime_settings
from bikipy.math.vector import dot_axis_1_1d, unit_vector

POINT_NAME_TO_INDEX = {"a": 0, "b": 1, "c": 2}


def _find_median_vector(row_vectors: Np2DArrayFp64) -> Np2DArrayFp64:
    """
    Computes the median point from a row vectors

    Median of each component -> combine medians to create median point. Note that this point doesn't exist

    Parameters
    ----------
    row_vectors: Np2DArrayFp64
        Array of row vectors

    Returns
    -------
    Np2DArrayFp64
    """
    return np.array([np.median(component) for component in row_vectors.T])


def clockwise_angel_2d(
    start_vector: Np2DArrayFp64,
    end_vector: Np2DArrayFp64,
) -> Np2DArrayFp64:
    """
    Computes the counterclockwise angle, [0, 2pi], from start to end in radians

    :param start_vector: Array of row vectors in which "the clock starts turning", counterclockwise
    :param end_vector: Array of row vectors in which the clock stops
    :type start_vector: Np2DArrayFp64
    :type end_vector: Np2DArrayFp64
    :return: counterclockwise angle between start and end vector per frame
    :rtype: Np2DArrayFp64

    >>> clockwise_angel_2d((1, 0), (0, 1))
    1.5707963267948966      # pi / 2.
    >>> clockwise_angel_2d((1, 0), (1, 0))
    0.0
    >>> clockwise_angel_2d((1, 0), (-1, 0))
    3.141592653589793       # pi"""

    start_vector = unit_vector(start_vector, force_1d=True)
    end_vector = unit_vector(end_vector, force_1d=True)

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

    dot_products = dot_axis_1_1d(end_vector, start_vector)
    angles = np.arctan2(np.abs(determinants), dot_products)
    angles[determinants < 0.0] = 2.0 * np.pi - angles[determinants < 0.0]
    return angles


def inner_angle(vector_set_a: Np2DArrayFp64, vector_set_b: Np2DArrayFp64):
    # Skip where either vector has NaN
    to_skip = np.any(np.isnan(vector_set_a), axis=1) | np.any(np.isnan(vector_set_b), axis=1)
    if runtime_settings.disable_numba:
        result = np.zeros(len(to_skip), dtype=np.float64)
        for i in np.where(~to_skip):
            result[i] = _inner_angle_compute(vector_set_a[i], vector_set_b[i])
        return result
    else:
        return _numba_inner_angle_loop(vector_set_a, vector_set_b, to_skip)


@njit(cache=True, parallel=True)
def _numba_inner_angle_loop(vector_set_a: Np2DArrayFp64, vector_set_b: Np2DArrayFp64, to_skip: Np1DArrayBool):
    result = np.zeros(len(vector_set_a), dtype=np.float64)
    for i in prange(len(vector_set_a)):
        if to_skip[i]:
            result[i] = np.nan
            continue
        result[i] = _inner_angle_compute(vector_set_a[i], vector_set_b[i])
    return result


def _inner_angle_compute(vector_a: Np2DArrayFp64, vector_b: Np2DArrayFp64) -> Np2DArrayFp64:
    minor = np.linalg.det(np.stack((vector_a, vector_b)))
    sign = 1 if minor == 0 else -np.sign(minor)

    dot_p = np.dot(vector_a, vector_b)
    dot_p = min(max(dot_p, -1.0), 1.0)

    return sign * np.arccos(dot_p)


def compute_angles_from_points_abc(
    row_vectors_point_a: Np2DArrayFp64,
    row_vectors_point_b: Np2DArrayFp64,
    row_vectors_point_c: Np2DArrayFp64,
    median_points: Optional[Sequence[str] | str] = None,
    method: str = "inner",
    degrees: bool = False,
) -> Np2DArrayFp64:
    """
    Computes the angle between three groups of vectors

    :param row_vectors_point_a: Array of row vectors
    :param row_vectors_point_b: Array of row vectors that is the joint between the two other groups of vectors
    :param row_vectors_point_c: Array of row vectors
    :param median_points: Anchor one or several points to their respective median. Information about median
                            computation in _find_median_vector()
    :param method: The method for computing angle, supported methods are inner; counterclockwise.
    :param degrees: If True, convert resulting angle data to degrees
    :type row_vectors_point_a: Np2DArrayFp64
    :type row_vectors_point_b: Np2DArrayFp64
    :type row_vectors_point_c: Np2DArrayFp64
    :type median_points: Iterable, str
    :type method: str
    :type degrees: bool
    :return: Angle per frame
    :rtype: Np2DArrayFp64
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


ANGLE_METHOD_TO_FUNC = {
    "inner": inner_angle,
    "counterclockwise": clockwise_angel_2d,
}


if not runtime_settings.disable_numba:
    _inner_angle_compute = njit(cache=True)(_inner_angle_compute)
