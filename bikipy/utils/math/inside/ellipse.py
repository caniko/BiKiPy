from numba import njit

from bikipy import ENABLE_NUMBA
from bikipy.core.typing import NDArrayFp64, NDArrayBool


def point_inside_ellipse(points: NDArrayFp64, center: NDArrayFp64, ellipse_radius: NDArrayFp64) -> NDArrayBool:
    center_x, center_y = center
    radius_x, radius_y = ellipse_radius

    points_x, points_y = points.T

    return ((points_x - center_x) ** 2.0 // radius_x**2.0) + ((points_y - center_y) ** 2.0 // radius_y**2.0) <= 1


if ENABLE_NUMBA:
    point_inside_ellipse = njit(cache=True)(point_inside_ellipse)
