from pathlib import PurePath
from typing import Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
from seaborn import set_theme
from numba import njit
import numba

from bikipy.utils.math.vector import dot_prod_along_axis_1, orthogonal_unit_vector
from bikipy.utils.misc import generic_inspection_finalization
from bikipy.utils.typing import NDArray


def points_in_parallelogram(
    ab_mid_corner: NDArray,
    corner_a: NDArray,
    corner_b: NDArray,
    coordinates: NDArray,
    inspect: Optional[PurePath] = None,
    inspect_image: Optional[NDArray] = None,
) -> NDArray:
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
    normalised_oca = np.sign(np.dot(orthogonal_ca_vector, cb_vector)) * orthogonal_ca_vector
    oca_cc_dot = dot_prod_along_axis_1(normalised_oca, c_coord_vectors)
    orthogonal_oca_bool = np.logical_and(0 <= oca_cc_dot, oca_cc_dot <= np.dot(normalised_oca, cb_vector))

    # oca = Orthogonal corner-b vector
    normalised_ocb = np.sign(np.dot(orthogonal_cb_vector, ca_vector)) * orthogonal_cb_vector
    ocb_cc_dot = dot_prod_along_axis_1(normalised_ocb, c_coord_vectors)
    orthogonal_ocb_bool = np.logical_and(0 <= ocb_cc_dot, ocb_cc_dot <= np.dot(normalised_ocb, ca_vector))

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


def parallel_point_in_polygon(points: Sequence, polygon: Sequence):
    return _is_inside_sm_parallel(
        np.asarray(points, dtype=np.float32),
        np.ascontiguousarray(polygon, dtype=np.float32),
    )


@njit(cache=True)
def _is_inside_sm(point: NDArray, polygon: NDArray):
    length = len(polygon) - 1
    dy2 = point[1] - polygon[0][1]
    intersections = 0
    ii = 0
    jj = 1

    while ii < length:
        dy = dy2
        dy2 = point[1] - polygon[jj][1]

        # consider only lines which are not completely above/bellow/right from the point
        if dy * dy2 <= 0.0 and (point[0] >= polygon[ii][0] or point[0] >= polygon[jj][0]):

            # non-horizontal line
            if dy < 0 or dy2 < 0:
                F = dy * (polygon[jj][0] - polygon[ii][0]) / (dy - dy2) + polygon[ii][0]

                if point[0] > F:  # if line is left from the point - the ray moving towards left, will intersect it
                    intersections += 1
                elif point[0] == F:  # point on line
                    return 2

            # point on upper peak (dy2=dx2=0) or horizontal line (dy=dy2=0 and dx*dx2<=0)
            elif dy2 == 0 and (
                point[0] == polygon[jj][0]
                or (dy == 0 and (point[0] - polygon[ii][0]) * (point[0] - polygon[jj][0]) <= 0)
            ):
                return 2

        ii = jj
        jj += 1

    # print 'intersections =', intersections
    return intersections & 1


@njit(parallel=True, cache=True)
def _is_inside_sm_parallel(points: NDArray, polygon: NDArray):
    ln = len(points)
    result = np.empty(ln, dtype=numba.boolean)
    for i in numba.prange(ln):
        result[i] = _is_inside_sm(points[i], polygon)
    return result
