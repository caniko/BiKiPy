from pandas import DataFrame
from warnings import warn
from typing import Union
import numpy as np


POINT_NAME_TO_INDEX = {"point_a": 0, "point_b": 1, "point_c": 2}


def _unit_vector(vector):
    """ Returns the unit vector of the vector. """
    return vector / np.linalg.norm(vector)


def _find_meridian_point(region_of_interest_data):
    return np.array([np.median(component) for component in region_of_interest_data.T])


def inner_clockwise_angel_2d(vector_a, vector_b) -> np.array:
    """ Computes the inner_angle in radians between two vectors

        :param vector_a: numpy array containing the location of per time
        :param vector_b: vector pair of vector_a

        >>> inner_clockwise_angel_2d((1, 0, 0), (0, 1, 0))
        1.5707963267948966      # pi / 2
        >>> inner_clockwise_angel_2d((1, 0, 0), (1, 0, 0))
        0.0
        >>> inner_clockwise_angel_2d((1, 0, 0), (-1, 0, 0))
        3.141592653589793       # pi
    """
    vector_1_u = np.apply_along_axis(_unit_vector, 1, np.asanyarray(vector_a))
    vector_2_u = np.apply_along_axis(_unit_vector, 1, np.asanyarray(vector_b))

    # Compute determinants and store them in a vertical stack
    determinants = np.array(
        [
            np.linalg.det(np.vstack((vector_1_u[i], vector_2_u[i])))
            for i in range(len(vector_1_u))
        ]
    )

    # dot = np.einsum("ij,ij->i", vector_1_u, vector_2_u)
    dot_prod = np.sum(vector_1_u * vector_2_u, axis=1)

    return np.pi - np.arctan2(determinants, dot_prod)


def angle_over_time(
    point_a: np.array,
    point_b: np.array,
    point_c: np.array,
    median_points: Union[str, list, None] = None,
    degrees: bool = False,
):
    points = [point_a, point_b, point_c]
    if any(point.dtype == "object" for point in points):
        warn("At least one of the arrays consists solely of NaN (Not a Number) objects")
        return np.full((points[0].size,), np.nan)

    if isinstance(median_points, list):
        for label in median_points:
            index = POINT_NAME_TO_INDEX[label]
            points[index] = _find_meridian_point(points[index])
    elif isinstance(median_points, str):
        index = POINT_NAME_TO_INDEX[median_points]
        points[index] = _find_meridian_point(points[index])
    else:
        msg = "median_points has to be list, string or None"
        ValueError(msg)

    computation = inner_clockwise_angel_2d(points[0] - points[1], points[1] - points[2])

    return np.rad2deg(computation) if degrees else computation


def dlc_angle_over_time(
    df: DataFrame, point_a_name: str, point_b_name: str, point_c_name: str, *args, **kwargs,
):
    from bikipy.utils.deeplabcut import (
        reduce_likelihoods,
        get_region_of_interest_data,
    )

    ordered_point_names = (point_a_name, point_b_name, point_c_name)
    likelihood = reduce_likelihoods(df, ordered_point_names)
    point_set = [
        get_region_of_interest_data(df, point_name)
        for point_name in ordered_point_names
    ]

    return {
        "Angle": angle_over_time(*point_set, *args, **kwargs),
        "Likelihood": likelihood,
    }
