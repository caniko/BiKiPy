from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np

from bikipy.math.vector import dot_prod_along_axis_1


def points_in_parallelogram(
    corner_point: Sequence, point_a: Sequence, point_b: Sequence, coordinates: Sequence,
    inspect_points: bool = False
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

    ca_cc_dot = dot_prod_along_axis_1(c_coord_vectors, ca_vector)
    cb_cc_dot = dot_prod_along_axis_1(c_coord_vectors, cb_vector)

    ca_cc_dot_booleans = np.logical_and(
        ca_cc_dot > 0, ca_cc_dot < np.linalg.norm(ca_vector) ** 2
    )
    cb_cc_dot_booleans = np.logical_and(
        cb_cc_dot > 0, cb_cc_dot < np.linalg.norm(cb_vector) ** 2
    )

    result = np.logical_and(ca_cc_dot_booleans, cb_cc_dot_booleans)

    if inspect_points:
        plt.plot(
            *np.append(corner_point, point_a).T, ".-r",
            *np.append(corner_point, point_b).T, ".-b"
        )
        plt.scatter(*coordinates[result].T)
        plt.scatter(*coordinates[np.logical_not(result)].T)
        plt.legend(("A", "B", "valid_points", "invalid_points"))
        plt.show()

    return result
