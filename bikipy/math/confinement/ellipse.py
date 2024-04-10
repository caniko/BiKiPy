from numba import njit
from pydantic_numpy.typing import Np1DArrayBool, NpNDArrayFp64

from bikipy import runtime_settings


def point_inside_ellipse(points: NpNDArrayFp64, center: NpNDArrayFp64, ellipse_radius: NpNDArrayFp64) -> Np1DArrayBool:
    center_x, center_y = center
    radius_x, radius_y = ellipse_radius

    points_x, points_y = points.T

    return ((points_x - center_x) ** 2.0 // radius_x**2.0) + ((points_y - center_y) ** 2.0 // radius_y**2.0) <= 1


if not runtime_settings.disable_numba:
    point_inside_ellipse = njit(cache=True)(point_inside_ellipse)
