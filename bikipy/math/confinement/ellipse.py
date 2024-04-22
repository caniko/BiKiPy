from numba import njit
from pydantic_numpy.typing import Np1DArrayBool, Np2DArrayFp64

from bikipy import runtime_settings


def point_inside_ellipse(points: Np2DArrayFp64, center: Np2DArrayFp64, ellipse_radius: Np2DArrayFp64) -> Np1DArrayBool:
    center_x, center_y = center
    radius_x, radius_y = ellipse_radius

    points_x, points_y = points.T

    return ((points_x - center_x) ** 2.0 // radius_x**2.0) + ((points_y - center_y) ** 2.0 // radius_y**2.0) <= 1


if not runtime_settings.disable_numba:
    point_inside_ellipse = njit(cache=True)(point_inside_ellipse)
