from logging import getLogger
from collections.abc import Sequence
from typing import SupportsFloat, Union

import numpy as np
from scipy.interpolate import interp1d

logger = getLogger(__name__)


def units_pixels_per_second_frame(units_per_pixel, fps):
    return units_per_pixel * fps


def displacement_per_frame(
    coordinate_sequence: Sequence[Sequence[SupportsFloat]],
) -> np.ndarray:
    coordinate_sequence = np.asarray(coordinate_sequence)
    magnitudes = np.linalg.norm(coordinate_sequence, axis=1)

    if not np.any((finite_indexes := np.where(np.isfinite(magnitudes))[0])):
        return np.abs(np.diff(magnitudes, axis=0))

    logger.debug(
        "Interpolating data as there are non-finite values in the location data"
    )

    try:
        f = interp1d(
            finite_indexes,
            magnitudes[finite_indexes],
            bounds_error=False,
            fill_value="extrapolate",
            copy=False,
            kind="cubic",
        )
    except ValueError:
        try:
            f = interp1d(
                finite_indexes,
                magnitudes[finite_indexes],
                bounds_error=False,
                fill_value="extrapolate",
                copy=False,
                kind="quadratic",
            )
        except ValueError:
            return np.abs(np.diff(magnitudes, axis=0))

    magnitudes = f(np.arange(magnitudes.size))

    return np.abs(np.diff(magnitudes, axis=0))


def displacement_mean_speed_acceleration(
    coordinate_sequence: Sequence[Sequence[SupportsFloat]],
    fps: SupportsFloat,
    length_unit_per_pixel: Union[SupportsFloat, None] = None,
    as_array: bool = True,
):
    coordinate_sequence = np.asarray(coordinate_sequence)
    fps = float(fps)
    length_unit_per_pixel = float(length_unit_per_pixel)

    displacement = displacement_per_frame(coordinate_sequence)
    delta_displacement = np.abs(np.diff(displacement, axis=0))

    total_displacement = np.sum(np.abs(displacement))
    if length_unit_per_pixel:
        total_displacement *= length_unit_per_pixel

    unit_convertor = units_pixels_per_second_frame(length_unit_per_pixel, fps)

    result = (
        total_displacement,
        # speed
        np.mean(displacement) * unit_convertor,
        # acceleration
        np.mean(delta_displacement) * unit_convertor,
    )
    return np.array(result) if as_array else result
