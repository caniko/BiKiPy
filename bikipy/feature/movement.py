from typing import Sequence, SupportsFloat, Union

import numpy as np


def units_pixels_per_second_frame(units_per_pixel, fps):
    return units_per_pixel * fps


def displacement_per_frame(
    location_sequence: Sequence[Sequence[SupportsFloat]],
) -> np.ndarray:
    location_sequence = np.asanyarray(location_sequence)
    return np.linalg.norm(np.diff(location_sequence, axis=0), axis=1)


def displacement_mean_speed_acceleration(
    location_sequence: Sequence[Sequence[SupportsFloat]],
    fps: SupportsFloat,
    unit_per_pixel: Union[SupportsFloat, None] = None,
    as_array: bool = True,
):
    location_sequence = np.asanyarray(location_sequence)
    fps = float(fps)
    unit_per_pixel = float(unit_per_pixel)

    displacement = displacement_per_frame(location_sequence)
    speed_per_frame = np.abs(np.diff(displacement, axis=0))
    acceleration_per_frame = np.abs(np.diff(speed_per_frame, axis=0))

    total_displacement = np.sum(displacement)
    if unit_per_pixel:
        total_displacement *= unit_per_pixel

    unit_convertor = units_pixels_per_second_frame(unit_per_pixel, fps)

    result = (
        total_displacement,
        np.mean(speed_per_frame) * unit_convertor,
        np.mean(acceleration_per_frame) * unit_convertor,
    )
    return np.array(result) if as_array else result
