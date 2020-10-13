from typing import Sequence

import numpy as np

from bikipy.utils.math import dot_prod_along_axis_1


def points_in_rectangle(
    corner_point: Sequence, point_a: Sequence, point_b: Sequence, coordinates: Sequence
):
    corner_point, point_a, point_b, coordinates = (
        np.asanyarray(corner_point),
        np.asanyarray(point_a),
        np.asanyarray(point_b),
        np.asanyarray(coordinates),
    )

    ca_vector = point_a - corner_point
    cb_vector = point_b - corner_point
    c_coord_vectors = coordinates - corner_point

    ca_cc_dot_booleans = (
        0
        < dot_prod_along_axis_1(ca_vector, c_coord_vectors)
        < 2 * np.linalg.norm(ca_vector)
    )
    cb_cc_dot_booleans = (
        0
        < dot_prod_along_axis_1(cb_vector, c_coord_vectors)
        < 2 * np.linalg.norm(cb_vector)
    )

    return np.logical_and(ca_cc_dot_booleans, cb_cc_dot_booleans)
