"""
Note that points in this context is the location of a region of interest across time.
"""
from typing import Sequence

import numpy as np


def compute_midpoint(point_1: Sequence, point_2: Sequence) -> np.ndarray:
    """
    Computes the point(s) in the middle of two points, midpoint(s).

    Parameters
    ----------
    point_1: np.ndarray
        Set of points part of the pair used for computing the midpoint(s)
    point_2
        Set of points part of the pair used for computing the midpoint(s)

    Returns
    -------
    np.ndarray; midpoint(s)
    """
    point_1, point_2 = np.asarray(point_1), np.asarray(point_2)

    if len(point_1.shape) == 1:
        point_1 = np.expand_dims(point_1, 0)
    if len(point_2.shape) == 1:
        point_2 = np.expand_dims(point_2, 0)

    point_1_vector_norms = np.apply_along_axis(np.linalg.norm, 1, point_1)
    point_2_vector_norms = np.apply_along_axis(np.linalg.norm, 1, point_2)

    # Vector location in which the respective vector has a larger size than the other
    i_greater_ii = point_1_vector_norms >= point_2_vector_norms
    # Opposite of the preceding
    ii_greater_i = np.logical_not(i_greater_ii)

    compute = np.zeros((ii_greater_i.size, 2))
    compute[i_greater_ii] = (
        point_2[i_greater_ii] + (point_1[i_greater_ii] - point_2[i_greater_ii]) / 2
    )

    compute[ii_greater_i] = (
        point_1[ii_greater_i] + (point_2[ii_greater_i] - point_1[ii_greater_i]) / 2
    )

    return compute


def recursive_midpoint(point_sets: Sequence) -> np.ndarray:
    """
    Compute midpoint(s) using last midpoint as first in the pair,
    and the upcoming point as the second in the pair in compute_midpoint.

    The first compute, where there is no midpoint, the last midpoint will be set to the
    first point in the sequence. This function could be interpreted as
    triangulating between three points when the length of the list is 3.

    Parameters
    ----------
    point_sets: np.ndarray
        Set of points used for computing the midpoint(s) recursively.

    Returns
    -------
    np.ndarray; midpoint(s).
    """
    midpoint = compute_midpoint(point_sets[0], point_sets[1])
    if len(midpoint) >= 2:
        for point_set in point_sets[2:]:
            midpoint = compute_midpoint(midpoint, point_set)

    return midpoint


def triangulate(point_1: Sequence, point_2: Sequence, point_3: Sequence) -> np.ndarray:
    """
    Midpoint between point_3, and the midpoint between point_1 and point_2

    Wrapper

    Parameters
    ----------
    point_1: np.ndarray
        Set of points part of the pair used for computing first midpoint(s).
    point_2: np.ndarray
        Set of points part of the pair used for computing first midpoint(s).
    point_3: np.ndarray
        Set of points part of the pair used for computing the second/last midpoint(s).

    Returns
    -------
    np.ndarray; triangulation between three points.
    """
    return recursive_midpoint((point_1, point_2, point_3))


def compute_from_dlc_df(df, point_group_names_set, min_likelihood: float = None):
    from bikipy.utils.deeplabcut import get_region_of_interest_data, reduce_likelihoods

    result = {}
    for group_subset_names in point_group_names_set:
        likelihood = reduce_likelihoods(df, group_subset_names)

        points = [
            get_region_of_interest_data(df, point_name)
            for point_name in group_subset_names
        ]

        compute_result = recursive_midpoint(points)

        if min_likelihood:
            compute_result[np.where(likelihood < min_likelihood)[0]] = np.nan

        result[f"mid-{'-'.join(group_subset_names)}"] = {
            "midpoint": compute_result.copy(),
            "likelihood": likelihood.copy(),
        }

    return result
