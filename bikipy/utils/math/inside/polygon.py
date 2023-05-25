from pathlib import PurePath
from typing import Any, Optional

import matplotlib.pyplot as plt
import numba
import numpy as np
from numba import njit
from pydantic import validate_arguments
from pydantic_numpy.dtype import NDArrayBool, NDArrayFp64
from seaborn import set_theme

from bikipy import runtime_settings
from bikipy.core.video import VideoMetadata
from bikipy.utils.image import axis_frame_imshow
from bikipy.utils.math.vector import dot_axis_1_1d, orthogonal_unit_vector
from bikipy.utils.plot.inspect import InspectArg, generic_inspection_finalization


def inaccurate_points_in_rectangle(
    ab_mid_corner: NDArrayFp64,
    corner_a: NDArrayFp64,
    corner_b: NDArrayFp64,
    coordinates: NDArrayFp64,
    inspect: Optional[PurePath],
    inspect_image: Optional[NDArrayFp64],
) -> np.ndarray[bool, bool]:
    """
    Algebraic solver for finding points contained inside the respective rectangle.

    Theoretical source: https://math.stackexchange.com/a/2643651/604035
    """

    ca_vector = corner_a - ab_mid_corner
    cb_vector = corner_b - ab_mid_corner
    c_coord_vectors = coordinates - ab_mid_corner

    if np.isclose(np.dot(ca_vector, cb_vector), 0.0):
        orthogonal_ca_vector = cb_vector
        orthogonal_cb_vector = ca_vector
    else:
        # Rectangle
        orthogonal_ca_vector = orthogonal_unit_vector(ca_vector)
        orthogonal_cb_vector = orthogonal_unit_vector(cb_vector)

    # oca = Orthogonal corner-a vector
    normalised_oca = np.sign(np.dot(orthogonal_ca_vector, cb_vector)) * orthogonal_ca_vector
    oca_cc_dot = dot_axis_1_1d(normalised_oca, c_coord_vectors)
    orthogonal_oca_bool = np.logical_and(0 <= oca_cc_dot, oca_cc_dot <= np.dot(normalised_oca, cb_vector))

    # oca = Orthogonal corner-b vector
    normalised_ocb = np.sign(np.dot(orthogonal_cb_vector, ca_vector)) * orthogonal_cb_vector
    ocb_cc_dot = dot_axis_1_1d(normalised_ocb, c_coord_vectors)
    orthogonal_ocb_bool = np.logical_and(0 <= ocb_cc_dot, ocb_cc_dot <= np.dot(normalised_ocb, ca_vector))

    boolean_index = orthogonal_oca_bool & orthogonal_ocb_bool

    if inspect:
        set_theme(style="darkgrid")
        fig, ax = plt.subplots()
        if inspect_image is not None:
            axis_frame_imshow(inspect_image, ax)

        ax.plot(*np.array((corner_a, ab_mid_corner)).T)
        ax.plot(*np.array((corner_b, ab_mid_corner)).T)

        ax.scatter(*coordinates[boolean_index].T)
        ax.scatter(*coordinates[~boolean_index].T)
        ax.legend(("A", "B", "valid_points", "invalid_points"))
        ax.set_title("Point in rectangle")

        generic_inspection_finalization(inspect)

    return boolean_index


@validate_arguments
def parallel_point_inside_polygon(
    points: NDArrayFp64,
    polygon: NDArrayFp64,
    merge_ends: bool = True,
    inspect_arg: InspectArg = False,
    video: Optional[VideoMetadata] = None,
    ax: Any = None,
    **inspect_kwargs,
) -> np.ndarray[bool, bool]:
    if merge_ends:
        polygon = np.append(polygon, np.expand_dims(polygon[0], 0), axis=0)

    result = is_inside_sm_parallel(
        np.asarray(points, dtype=np.float64),
        np.ascontiguousarray(polygon, dtype=np.float64),
    )

    if inspect_arg or ax is not None:
        if video:
            fig, ax = video.subplot()

            points = video.prepare_coordinates_for_plotting(points)

            ax.plot(*np.vstack(video.prepare_coordinates_for_plotting(polygon)).T, label="Polygon")
            ax.scatter(*points[result].T, label="Inside")
            ax.scatter(*points[~result].T, label="Outside")
        else:
            if ax is None:
                fig, ax = plt.subplots()

            ax.plot(*np.vstack(polygon).T, label="Polygon")
            ax.scatter(*points[result].T, label="Inside")
            ax.scatter(*points[~result].T, label="Outside")

        ax.legend()
        generic_inspection_finalization(inspect_arg, **inspect_kwargs)

    return result


def _is_inside_sm(point: NDArrayFp64, polygon: NDArrayFp64):
    """
    https://stackoverflow.com/a/66189882
    https://github.com/sasamil/PointInPolygon_Py/blob/master/pointInside.py
    """
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


if runtime_settings.disable_numba:
    is_inside_sm = _is_inside_sm

    def is_inside_sm_parallel(points: NDArrayFp64, polygon: NDArrayFp64) -> np.ndarray[bool, bool]:
        ln = len(points)
        result = np.empty(ln, dtype=bool)
        for i in range(ln):
            result[i] = is_inside_sm(points[i], polygon)
        return result

else:
    is_inside_sm = njit(nogil=True, cache=True)(_is_inside_sm)

    @njit(parallel=True, cache=True)
    def is_inside_sm_parallel(points: NDArrayFp64, polygon: NDArrayFp64) -> np.ndarray[bool, bool]:
        ln = len(points)
        result = np.empty(ln, dtype=numba.boolean)
        for i in numba.prange(ln):
            result[i] = is_inside_sm(points[i], polygon)
        return result
