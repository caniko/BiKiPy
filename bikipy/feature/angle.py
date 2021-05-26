from typing import AnyStr, Sequence, Union
from warnings import warn

import numpy as np
from pandas.core.frame import DataFrame as DataFrameType

from bikipy.math.statistics import feature_scale
from bikipy.math.vector import dot_prod_along_axis_1, unit_vector

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


def counter_clockwise_angel_2d(start_vector, end_vector) -> np.ndarray:
    """
    Computes the clockwise inner_angle in radians between two vectors

    Parameters
    ----------
    start_vector: np.ndarray
        Array of row vectors in which "the clock starts turning" counter clockwise
    end_vector: np.ndarray
        Array of row vectors in which the clock stops

    Returns
    -------
    np.ndarray

    >>> counter_clockwise_angel_2d((1, 0), (0, 1))
    1.5707963267948966      # pi / 2.
    >>> counter_clockwise_angel_2d((1, 0), (1, 0))
    0.0
    >>> counter_clockwise_angel_2d((1, 0), (-1, 0))
    3.141592653589793       # pi
    """

    start_unit_vector = unit_vector(start_vector, force_1_dim=True)
    end_unit_vector = unit_vector(end_vector, force_1_dim=True)

    # Compute determinants and store them in a vertical stack
    determinants = np.array(
        [
            np.linalg.det(np.vstack((start_unit_vector[i], end_unit_vector[i])))
            for i in range(len(start_unit_vector))
        ]
    )

    dot_prod = dot_prod_along_axis_1(start_unit_vector, end_unit_vector)

    return np.pi - np.arctan2(determinants, dot_prod)


def compute_angles_from_vectors(
    row_vectors_point_a: np.ndarray,
    row_vectors_point_b: np.ndarray,
    row_vectors_point_c: np.ndarray,
    median_points: Union[AnyStr, Sequence, None] = None,
    feature_scale_data: bool = False,
    feature_scale_min_max: Union[Sequence, None] = None,
    degrees: bool = False,
):
    """
    Computes the angle between three groups of vectors

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
    feature_scale_data: bool; default False
        If True, data will be scaled based on minimum and maximum of data
    feature_scale_min_max: iterable(min, max); default None
        Optional override of minimum and maximum used for feature scaling
    degrees: bool; default False
        If True, convert resulting angle data to degrees

    Returns
    -------
    np.ndarray
    """

    points = [
        np.asarray(row_vectors_point_a),
        np.asarray(row_vectors_point_b),
        np.asarray(row_vectors_point_c),
    ]
    if any(point.dtype == "object" for point in points):
        warn("At least one of the arrays consists solely of NaN (Not a Number) objects")
        return np.full((points[0].size,), np.nan)

    if not median_points:
        pass
    elif isinstance(median_points, str):
        index = POINT_NAME_TO_INDEX[median_points]
        points[index] = _find_median_vector(points[index])
    else:
        try:
            for label in median_points:
                index = POINT_NAME_TO_INDEX[label]
                points[index] = _find_median_vector(points[index])
        except KeyError as e:
            msg = "When median_points is an Iterable it must store either a, b and/or c"
            raise KeyError(msg) from e
        except TypeError as e:
            msg = "median_points has to be list, string or None"
            raise ValueError(msg) from e

    computation = counter_clockwise_angel_2d(
        points[1] - points[0], points[2] - points[1]  # AB Vector  # BC Vector
    )

    if feature_scale_data:
        if feature_scale_min_max:
            try:
                computation = feature_scale(computation, *feature_scale_min_max)
            except TypeError as e:
                msg = "feature_scale_data must either be (min, max) or bool"
                raise ValueError(msg) from e
        else:
            computation = feature_scale(computation)

    elif degrees:
        computation = np.rad2deg(computation)

    return computation


def dlc_compute_angles_from_vectors(
    df: DataFrameType,
    point_a_name: AnyStr,
    point_b_name: AnyStr,
    point_c_name: AnyStr,
    *args,
    **kwargs,
):
    """
    compute_angles_from_vectors wrapper for DataFrames generated from DeepLabCut 2d result files

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
    from bikipy.utils.deeplabcut import get_region_of_interest_data, reduce_likelihoods

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
