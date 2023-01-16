"""
Note that points in this context is the location of a region of interest across time.
"""
import numpy as np
from pydantic import validate_arguments
from pydantic_numpy.dtype import NDArrayFp64


@validate_arguments
def compute_midpoint(point_1: NDArrayFp64, point_2: NDArrayFp64, midpoint_quotient: float = 2.0) -> NDArrayFp64:
    """
    Computes the point(s) between two points, midpoint(s), with respect to the index.

    :param point_1: Set of points part of the pair used for computing the midpoint(s)
    :param point_2: Set of points part of the pair used for computing the midpoint(s)
    :param midpoint_quotient:
    :return: midpoint(s)
    :rtype: NDArrayFp64
    """
    if len(point_1.shape) == 1:
        point_1 = np.expand_dims(point_1, 0)
    if len(point_2.shape) == 1:
        point_2 = np.expand_dims(point_2, 0)

    # Vector location in which the respective vector has a larger size than the other
    # point_1_vector_norms >= point_2_vector_norms
    i_greater_equals_ii = np.linalg.norm(point_1, axis=1) >= np.linalg.norm(point_2, axis=1)
    ii_greater_i = ~i_greater_equals_ii

    compute = np.zeros((ii_greater_i.size, 2))
    compute[i_greater_equals_ii] = (
        point_2[i_greater_equals_ii] + (point_1[i_greater_equals_ii] - point_2[i_greater_equals_ii]) / midpoint_quotient
    )
    compute[ii_greater_i] = point_1[ii_greater_i] + (point_2[ii_greater_i] - point_1[ii_greater_i]) / midpoint_quotient

    return compute


def recursive_midpoint(*point_sets: NDArrayFp64, midpoint_quotient: float = 2.0) -> NDArrayFp64:
    """
    Compute midpoint(s) using last midpoint as first in the pair,
    and the upcoming point as the second in the pair in compute_midpoint.

    The first compute, where there is no midpoint, the last midpoint will be set to the
    first point in the sequence. This function could be interpreted as
    triangulating between three points when the length of the list is 3.

    :param point_sets: Iterable of points used for computing the midpoint(s) recursively.
    :param midpoint_quotient:
    :return: midpoint(s)
    :rtype: NDArrayFp64
    """
    try:
        midpoint = compute_midpoint(point_sets[0], point_sets[1])
    except IndexError as e:
        msg = (
            f"There needs to be at least 2 point sets to compute midpoints recursively, there is only {len(point_sets)}"
        )
        raise ValueError(msg) from e

    try:
        for point_set in point_sets[2:]:
            midpoint = compute_midpoint(midpoint, point_set, midpoint_quotient)
    except IndexError:
        pass

    return midpoint
