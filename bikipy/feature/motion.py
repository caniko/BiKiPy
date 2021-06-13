from collections.abc import Sequence
from logging import getLogger
from typing import Union

import numpy as np
import pandas as pd

from bikipy.math.calculus import absolute_derivative

logger = getLogger(__name__)


def units_pixels_per_second_frame(
    units_per_pixel: Union[float, int], fps: Union[float, int]
):
    return units_per_pixel * fps


def displacement_per_frame(
    coordinate_sequence: Sequence[Sequence[float]], interpolation_method: str = "akima"
) -> np.ndarray:
    """
    Compute the absolute displacement of the given point from its coordinates across frames.
    The values that are undefined, or "not a number" (NaN), on the tails are removed, and the
    undefined values that border defined values are interpolate.

    Parameters
    ----------
    coordinate_sequence
        The respective coordinate sequence
    interpolation_method

    Returns
    -------
    np.ndarray with pixel displacement per frame
    """
    if not np.any(np.isnan((
        magnitudes := np.linalg.norm(coordinate_sequence, axis=1))
    )):
        return absolute_derivative(magnitudes)

    logger.debug(
        "Interpolating data as there are non-finite values in the location data"
    )

    magnitudes_series = pd.Series(magnitudes)
    magnitudes_series.interpolate(
        method=interpolation_method,
        limit_direction="both",
        limit_area="inside",
        inplace=True,
    )
    magnitudes_series.dropna(inplace=True)

    return absolute_derivative(magnitudes_series.values)


def total_displacement_median_speed_acceleration(
    coordinate_sequence: Sequence[Sequence[float]],
    unit_per_pixel: float,
    fps: float,
) -> tuple:
    """

    Parameters
    ----------
    coordinate_sequence
    unit_per_pixel
    fps

    Returns
    -------
    (total displacement, speed per frame, acceleration per frame)
    """
    displacement = displacement_per_frame(coordinate_sequence) * unit_per_pixel
    if np.any(displacement):
        return (
            np.sum(displacement),
            np.nanmedian((speed := absolute_derivative(displacement) * fps)),
            np.nanmedian(absolute_derivative(speed)),
        )
    else:
        return 0, 0, 0


class Motion:
    def __init__(
        self,
        coordinate_sequence: Sequence[Sequence[float]],
        unit_per_pixel: float,
        fps: float,
    ):
        self.metric_displacement_per_frame = (
            displacement_per_frame(coordinate_sequence) * unit_per_pixel
        )

        self.total_displacement = np.nansum(self.metric_displacement_per_frame)
        if self.total_displacement:
            self.speed = absolute_derivative(
                self.metric_displacement_per_frame) * fps
            self.median_speed = np.nanmedian(self.speed)

            self.acceleration = absolute_derivative(self.speed)
            self.median_acceleration = np.nanmedian(self.acceleration)
        else:
            self.speed = None
            self.median_speed = None

            self.acceleration = None
            self.median_acceleration = None

    def to_list(self):
        return [
            self.total_displacement,
            self.median_speed,
            self.median_acceleration,
        ]
