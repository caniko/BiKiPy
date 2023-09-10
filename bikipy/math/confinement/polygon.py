import numba
import numpy as np
from numba import njit
from pydantic import validate_call
from pydantic_numpy.typing import NpNDArrayBool, NpNDArrayFp64

from bikipy import runtime_settings


@validate_call
def parallel_point_inside_polygon(
    points: NpNDArrayFp64, polygon: NpNDArrayFp64, merge_ends: bool = True
) -> NpNDArrayBool:
    if merge_ends:
        polygon = np.append(polygon, np.expand_dims(polygon[0], 0), axis=0)

    return is_inside_sm_parallel(
        np.ascontiguousarray(points, dtype=np.float64),
        np.ascontiguousarray(polygon, dtype=np.float64),
    )


def is_inside_sm(point: NpNDArrayFp64, polygon: NpNDArrayFp64):
    """
    Does not work when the intersection is on (0, 0); very rare case

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

    def is_inside_sm_parallel(points: NpNDArrayFp64, polygon: NpNDArrayFp64) -> NpNDArrayBool:
        ln = len(points)
        result = np.empty(ln, dtype=bool)
        for i in range(ln):
            result[i] = is_inside_sm(points[i], polygon)
        return result

else:
    is_inside_sm = njit(nogil=True, cache=True)(is_inside_sm)

    @njit(parallel=True, cache=True)
    def is_inside_sm_parallel(points: NpNDArrayFp64, polygon: NpNDArrayFp64) -> NpNDArrayBool:
        ln = len(points)
        result = np.empty(ln, dtype=numba.boolean)
        for i in numba.prange(ln):
            result[i] = is_inside_sm(points[i], polygon)
        return result
