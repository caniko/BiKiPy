from logging import getLogger
from collections.abc import Sequence

import numpy as np
import pandas as pd

logger = getLogger(__name__)


def units_pixels_per_second_frame(units_per_pixel: float, fps: float):
    return units_per_pixel * fps


def displacement_per_frame(coordinate_sequence: Sequence[Sequence[float]]) -> np.ndarray:
    """
    Compute the absolute displacement of the given point from its coordinates across frames.
    The values that are undefined, or "not a number" (NaN), on the tails are removed, and the
    ones that border defined values are interpolate.

    Parameters
    ----------
    coordinate_sequence
        The respective coordinate sequence

    Returns
    -------
    np.ndarray with g
    """
    if np.all((
        magnitudes := np.linalg.norm(coordinate_sequence, axis=1)
    )):
        return np.abs(np.diff(magnitudes, axis=0))

    logger.debug(
        "Interpolating data as there are non-finite values in the location data"
    )

    magnitudes_series = pd.Series(magnitudes)
    magnitudes_series.interpolate(method="akima", limit_direction="both", limit_area="inside", inplace=True)
    magnitudes_series.dropna(inplace=True)

    return np.abs(np.diff(magnitudes_series.values, axis=0))


def displacement_mean_speed_acceleration(
    coordinate_sequence: Sequence[Sequence[float]],
    fps: float,
    length_unit_per_pixel: float,
) -> tuple[float, np.ndarray, np.ndarray]:
    """

    Parameters
    ----------
    coordinate_sequence
    fps
    length_unit_per_pixel

    Returns
    -------

    """
    length_unit_per_pixel = float(length_unit_per_pixel)

    displacement = displacement_per_frame(coordinate_sequence) * length_unit_per_pixel
    speed = np.abs(np.diff(displacement, axis=0)) * fps

    return (
        np.sum(displacement),
        speed,
        np.abs(np.diff(speed, axis=0))  # acceleration
    )
