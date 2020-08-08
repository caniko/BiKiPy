from pandas import DataFrame
from warnings import warn
from typing import Union
import numpy as np


POINT_NAME_TO_INDEX = {
    "row_vectors_point_a": 0,
    "row_vectors_point_b": 1,
    "row_vectors_point_c": 2,
}


def _unit_vector(row_vectors) -> np.ndarray:
    """ Computes unit vector, i.e. vector/<norm of the vector>

    Parameters
    ----------
    row_vectors: np.ndarray-like
        Array of row vector(s)

    Returns
    -------
    np.ndarray
    """
    return row_vectors / np.linalg.norm(row_vectors)


def _find_median_vector(row_vectors: np.ndarray) -> np.ndarray:
    """ Computes the median point from a row vectors

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


def inner_clockwise_angel_2d(vector_a, vector_b) -> np.ndarray:
    """ Computes the clockwise inner_angle in radians between two vectors
    
    Parameters
    ----------
    vector_a: np.ndarray
        Array of row vectors
    vector_b: np.ndarray
        Array of row vectors

    Returns
    -------
    np.ndarray
    
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


def compute_angles_from_vectors(
    row_vectors_point_a: np.ndarray,
    row_vectors_point_b: np.ndarray,
    row_vectors_point_c: np.ndarray,
    median_points: Union[str, list, None] = None,
    degrees: bool = False,
):
    """ Compute the angle between three groups of vectors

    Computed with respect to their index in their respective store.

    Parameters
    ----------
    row_vectors_point_a: np.ndarray
        Array of row vectors
    row_vectors_point_b: np.ndarray
        Array of row vectors that is the joint between the two other groups of vectors
    row_vectors_point_c: np.ndarray
        Array of row vectors
    median_points: str, list; optional
        Anchor one or several row vectors to their respective median. Information about median computation in _find_median_vector()
    degrees: bool; default False
        If True, convert resulting angle data to degrees

    Returns
    -------
    np.ndarray
    """

    points = [row_vectors_point_a, row_vectors_point_b, row_vectors_point_c]
    if any(point.dtype == "object" for point in points):
        warn("At least one of the arrays consists solely of NaN (Not a Number) objects")
        return np.full((points[0].size,), np.nan)

    if isinstance(median_points, list):
        for label in median_points:
            index = POINT_NAME_TO_INDEX[label]
            points[index] = _find_median_vector(points[index])
    elif isinstance(median_points, str):
        index = POINT_NAME_TO_INDEX[median_points]
        points[index] = _find_median_vector(points[index])
    else:
        msg = "median_points has to be list, string or None"
        ValueError(msg)

    computation = inner_clockwise_angel_2d(points[0] - points[1], points[1] - points[2])

    return np.rad2deg(computation) if degrees else computation


def dlc_compute_angles_from_vectors(
    df: DataFrame,
    point_a_name: str,
    point_b_name: str,
    point_c_name: str,
    *args,
    **kwargs,
):
    """ compute_angles_from_vectors wrapper for DataFrames generated from DeepLabCut 2d result files

    Parameters
    ----------
    df: DataFrame
        Data from DeepLabCut ingested as a pd.DataFrame
    point_a_name: str
        Name of vector group
    point_b_name: str
        Name of the vector that is the joint between the two other groups
    point_c_name: str
        Name of vector group
    args
        Arguments for compute_angles_from_vectors
    kwargs
        Keyword arguments for compute_angles_from_vectors

    Returns
    -------
    dict: {Angle, Likelihood}
    """
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
        "Angle": compute_angles_from_vectors(*point_set, *args, **kwargs),
        "Likelihood": likelihood,
    }
