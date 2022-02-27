from pathlib import PurePath
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from seaborn import set_theme

from bikipy.math.vector import dot_prod_along_axis_1, orthogonal_unit_vector
from bikipy.utils.misc import generic_inspection_finalization


def points_in_parallelogram(
    ab_mid_corner: np.ndarray,
    corner_a: np.ndarray,
    corner_b: np.ndarray,
    coordinates: np.ndarray,
    inspect: Optional[PurePath] = None,
    inspect_image: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Algebraic solver for finding points contained inside the respective parallelogram.

    Theoretical source: https://math.stackexchange.com/a/2643651/604035
    """

    ca_vector = corner_a - ab_mid_corner
    cb_vector = corner_b - ab_mid_corner
    c_coord_vectors = coordinates - ab_mid_corner

    if np.isclose(np.dot(ca_vector, cb_vector), 0.0):
        orthogonal_ca_vector = cb_vector
        orthogonal_cb_vector = ca_vector
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

    boolean_index = orthogonal_oca_bool & orthogonal_ocb_bool

    if inspect:
        set_theme(style="darkgrid")
        fig, ax = plt.subplots()
        if inspect_image is not None:
            ax.imshow(inspect_image)

        ax.plot(*np.array((corner_a, ab_mid_corner)).T)
        ax.plot(*np.array((corner_b, ab_mid_corner)).T)

        ax.scatter(*coordinates[boolean_index].T)
        ax.scatter(*coordinates[~boolean_index].T)
        ax.legend(("A", "B", "valid_points", "invalid_points"))
        ax.set_title("Point in parallelogram")

        generic_inspection_finalization(inspect)

    return boolean_index
