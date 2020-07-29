import pandas as pd
import numpy as np

from bikipy.utils import compute_nan_ratio, validate_nan_ratio


def _unit_vector(vector):
    """ Returns the unit vector of the vector. """
    return vector / np.linalg.norm(vector)


def clockwise_2d(vector_a, vector_b) -> np.array:
    """ Returns the inner_angle in radians between two vectors

        :param vector_a: numpy array containing the location of per time
        :param vector_b: vector pair of vector_a

        >>> clockwise_2d((1, 0, 0), (0, 1, 0))
        1.5707963267948966
        >>> clockwise_2d((1, 0, 0), (1, 0, 0))
        0.0
        >>> clockwise_2d((1, 0, 0), (-1, 0, 0))
        3.141592653589793
    """
    vector_1_u = np.apply_along_axis(_unit_vector, 1, np.asarray(vector_a))
    vector_2_u = np.apply_along_axis(_unit_vector, 1, np.asarray(vector_b))

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


def angle_over_time(df: pd.DataFrame, point_a, point_b, point_c):
    position = {
        point: df.loc[:, [(point, "x"), (point, "y")]].values
        for point in (point_a, point_b, point_c)
    }

    ab_vec = position[point_a] - position[point_b]
    bc_vec = position[point_b] - position[point_c]

    nan_ratio = compute_nan_ratio((ab_vec, bc_vec))
    validate_nan_ratio(nan_ratio)

    angles = clockwise_2d(ab_vec, bc_vec)
    accuracy_score = (
        df.loc[:, [(point_a, "likelihood")]].values
        + df.loc[:, [(point_b, "likelihood")]].values
        + df.loc[:, [(point_c, "likelihood")]].values
    ) / 3

    return {"Angles": angles, "Accuracy Score": accuracy_score.T[0]}
