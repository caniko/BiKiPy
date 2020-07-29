"""
Note that points in this context is the location of a region of interest across time.
"""


from typing import Sequence
import numpy as np

from bikipy.utils import compute_nan_ratio, validate_nan_ratio


def compute_midpoint(point_1, point_2) -> np.array:
    point_1, point_2 = np.asarray(point_1), np.asarray(point_2)

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


def recursive_midpoint(points: Sequence) -> np.array:
    """ Compute midpoints using last midpoint as point 1, and the next point
    as point 2 in compute_midpoint.

    The first compute, where there is no midpoint, the last midpoint will be set to the
    first point in the sequence.

    For example:
    This function could be interpreted as triangulating between three points
    when the length of the list is 3.

    :param points:
    :return:
    """
    midpoint = points[0]
    for i in range(1, len(points)):
        midpoint = compute_midpoint(midpoint, points[i])
    return midpoint


def triangulate(point_1, point_2, point_3) -> np.array:
    """ Midpoint between point_3, and the midpoint between point_1 and point_2

    :param point_1: Set of points used for computing first midpoint
    :param point_2: Set of points used for computing first midpoint
    :param point_3: Set of points used for computing the last midpoint
    :return: triangulation between three points
    """
    return recursive_midpoint((point_1, point_2, point_3))


def compute_from_dlc_df(df, point_group_names_set, min_likelihood: float = 0.95):
    result = {}
    for group_subset_names in point_group_names_set:
        likelihood = np.array(
            [
                df.loc[:, [(point_name, "likelihood")]].values
                for point_name in group_subset_names
            ]
        ) / len(group_subset_names)

        points = []
        for point_name in group_subset_names:
            rem = df.loc[:, [(point_name, "x"), (point_name, "y")]].values
            nan_ratio = compute_nan_ratio(rem)
            validate_nan_ratio(nan_ratio)

            points.append(rem)

        points = [
            df.loc[:, [(point_name, "x"), (point_name, "y")]].values
            for point_name in group_subset_names
        ]

        compute_result = recursive_midpoint(points)

        if min_likelihood:
            compute_result[np.where(likelihood < min_likelihood)[0]] = np.nan

        result[f"mid-{'-'.join(group_subset_names)}"] = {
            "midpoint": compute_result,
            "likelihood": likelihood[0],
        }

    return result
