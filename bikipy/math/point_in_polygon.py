from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np
from seaborn import set_theme

from bikipy.math.vector import dot_prod_along_axis_1


def points_in_parallelogram(
    ab_mid_corner: Sequence,
    corner_a: Sequence,
    corner_b: Sequence,
    coordinates: Sequence,
    inspect_points: bool = False,
) -> np.ndarray:
    """
    Algebraic solver for finding points located inside a parallelogram.

    Parameters
    ----------
    ab_mid_corner
    corner_a
    corner_b
    coordinates
    inspect_points

    Returns
    -------

    """
    ab_mid_corner, corner_a, corner_b, coordinates = (
        np.asarray(ab_mid_corner),
        np.asarray(corner_a),
        np.asarray(corner_b),
        np.asarray(coordinates),
    )

    assert all(
        coord_array.shape[-1] == 2
        for coord_array in (ab_mid_corner, corner_a, corner_b, coordinates)
    ), "Coordinate data must be 2 dimensional"

    ca_vector = corner_a - ab_mid_corner
    cb_vector = corner_b - ab_mid_corner
    c_coord_vectors = coordinates - ab_mid_corner

    ca_cc_dot = dot_prod_along_axis_1(c_coord_vectors, ca_vector)
    cb_cc_dot = dot_prod_along_axis_1(c_coord_vectors, cb_vector)

    if np.isclose(np.dot(ca_vector, cb_vector), 0.0):
        ca_cc_dot_booleans = np.logical_and(
            ca_cc_dot > 0, ca_cc_dot < np.linalg.norm(ca_vector) ** 2
        )

        cb_cc_dot_booleans = np.logical_and(
            cb_cc_dot > 0, cb_cc_dot < np.linalg.norm(cb_vector) ** 2
        )

        result = np.logical_and(ca_cc_dot_booleans, cb_cc_dot_booleans)
    else:
        pass

    if inspect_points:
        set_theme(style="darkgrid")
        fig, ax = plt.subplots()
        for point in (corner_a, ab_mid_corner, corner_b):
            ax.scatter(*point.T)

        ax.scatter(*coordinates[result].T)
        ax.scatter(*coordinates[~result].T)
        ax.legend(("A", "corner_points", "B", "valid_points", "invalid_points"))
        ax.set_title("Point in parallelogram")

        plt.show()

    return result
