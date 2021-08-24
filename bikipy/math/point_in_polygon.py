from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np
from numba import jit
from seaborn import set_theme

from bikipy.math.vector import dot_prod_along_axis_1, orthogonal_unit_vector


@jit
def points_in_parallelogram(
    ab_mid_corner: Sequence,
    corner_a: Sequence,
    corner_b: Sequence,
    coordinates: Sequence,
    inspect_points: bool = False,
) -> np.ndarray:
    """
    Algebraic solver for finding points contained inside the respective parallelogram.

    Theoretical source: https://math.stackexchange.com/a/2643651/604035

    :param ab_mid_corner:
    :param corner_a:
    :param corner_b:
    :param coordinates:
    :param inspect_points:
    :type ab_mid_corner: np.ndarray
    :type corner_a: np.ndarray
    :type corner_b: np.ndarray
    :type coordinates: np.ndarray
    :type inspect_points: bool
    :return:
    :rtype np.ndarray
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

    if np.isclose(np.dot(ca_vector, cb_vector), 0.0):
        # Rectangle
        orthogonal_ca_vector = cb_vector
        orthogonal_cb_vector = ca_vector

        # ca_cc_dot = dot_prod_along_axis_1(c_coord_vectors, ca_vector)
        # cb_cc_dot = dot_prod_along_axis_1(c_coord_vectors, cb_vector)
        #
        # ca_cc_dot_booleans = np.logical_and(
        #     0 < ca_cc_dot, ca_cc_dot < np.linalg.norm(ca_vector) ** 2
        # )
        #
        # cb_cc_dot_booleans = np.logical_and(
        #     0 < cb_cc_dot, cb_cc_dot < np.linalg.norm(cb_vector) ** 2
        # )
        #
        # result = ca_cc_dot_booleans & cb_cc_dot_booleans
    else:
        # Parallelogram
        orthogonal_ca_vector = orthogonal_unit_vector(ca_vector)
        orthogonal_cb_vector = orthogonal_unit_vector(cb_vector)

    # oca = Orthogonal corner-a vector
    normalised_oca = (
        np.sign(np.dot(orthogonal_ca_vector, cb_vector)) * orthogonal_ca_vector
    )
    oca_cc_dot = dot_prod_along_axis_1(normalised_oca, c_coord_vectors)
    orthogonal_oca_bool = np.logical_and(
        0 <= oca_cc_dot, oca_cc_dot <= np.dot(normalised_oca, cb_vector)
    )

    # oca = Orthogonal corner-b vector
    normalised_ocb = (
        np.sign(np.dot(orthogonal_cb_vector, ca_vector)) * orthogonal_cb_vector
    )
    ocb_cc_dot = dot_prod_along_axis_1(normalised_ocb, c_coord_vectors)
    orthogonal_ocb_bool = np.logical_and(
        0 <= ocb_cc_dot, ocb_cc_dot <= np.dot(normalised_ocb, ca_vector)
    )

    result = orthogonal_oca_bool & orthogonal_ocb_bool

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
