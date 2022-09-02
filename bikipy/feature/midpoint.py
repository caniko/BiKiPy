"""
Note that points in this context is the location of a region of interest across time.
"""
from typing import Sequence

import numpy as np
from pydantic_numpy.dtype import NDArrayFp64


def compute_midpoint(point_1: NDArrayFp64, point_2: NDArrayFp64) -> NDArrayFp64:
    """
    Computes the point(s) between two points, midpoint(s), with respect to the index.

    :param point_1: Set of points part of the pair used for computing the midpoint(s)
    :param point_2: Set of points part of the pair used for computing the midpoint(s)
    :type point_1: NDArrayFp64
    :type point_2: NDArrayFp64
    :return: midpoint(s)
    :rtype: NDArrayFp64
    """
    point_1, point_2 = np.asarray(point_1), np.asarray(point_2)

    if len(point_1.shape) == 1:
        point_1 = np.expand_dims(point_1, 0)
    if len(point_2.shape) == 1:
        point_2 = np.expand_dims(point_2, 0)

    point_1_vector_norms = np.apply_along_axis(np.linalg.norm, 1, point_1)
    point_2_vector_norms = np.apply_along_axis(np.linalg.norm, 1, point_2)

    # Vector location in which the respective vector has a larger size than the other
    i_greater_equals_ii = point_1_vector_norms >= point_2_vector_norms
    # Opposite of the preceding
    ii_greater_i = ~i_greater_equals_ii

    compute = np.zeros((ii_greater_i.size, 2))
    compute[i_greater_equals_ii] = (
        point_2[i_greater_equals_ii] + (point_1[i_greater_equals_ii] - point_2[i_greater_equals_ii]) / 2.0
    )

    compute[ii_greater_i] = point_1[ii_greater_i] + (point_2[ii_greater_i] - point_1[ii_greater_i]) / 2.0

    return compute


def recursive_midpoint(point_sets: Sequence[NDArrayFp64]) -> NDArrayFp64:
    """
    Compute midpoint(s) using last midpoint as first in the pair,
    and the upcoming point as the second in the pair in compute_midpoint.

    The first compute, where there is no midpoint, the last midpoint will be set to the
    first point in the sequence. This function could be interpreted as
    triangulating between three points when the length of the list is 3.

    :param point_sets: Set of points used for computing the midpoint(s) recursively.
    :type point_sets: Sequence
    :return: midpoint(s)
    :rtype: NDArrayFp64
    """
    midpoint = compute_midpoint(point_sets[0], point_sets[1])
    try:
        for point_set in point_sets[2:]:
            midpoint = compute_midpoint(midpoint, point_set)
    except IndexError:
        pass

    return midpoint
