from typing import Union, Sequence, SupportsFloat

import numpy as np

from bikipy.utils.video import seconds_from_frames


def displacement(location_sequence: Sequence[Sequence[SupportsFloat]]) -> np.ndarray:
    location_sequence = np.asanyarray(location_sequence)
    return np.linalg.norm(np.diff(location_sequence, axis=0), axis=1)


def displacement_mean_speed_acceleration(
    location_sequence: Sequence[Sequence[SupportsFloat]],
    fps: SupportsFloat,
    unit_per_pixel: Union[SupportsFloat, None] = None,
    as_array: bool = True,
):
    def mean_of_feature_per_fps(feature: np.ndarray):
        mean_feature = np.sum(feature) / total_seconds

        if unit_per_pixel:
            mean_feature *= unit_per_pixel

        return mean_feature

    location_sequence = np.asanyarray(location_sequence)
    fps = float(fps)
    unit_per_pixel = float(unit_per_pixel)

    displacement_per_frame = displacement(location_sequence)
    speed_per_frame = np.abs(np.diff(displacement_per_frame, axis=0))
    acceleration_per_frame = np.abs(np.diff(speed_per_frame, axis=0))

    total_displacement = np.sum(displacement_per_frame)
    if unit_per_pixel:
        total_displacement *= unit_per_pixel

    total_seconds = seconds_from_frames(fps, location_sequence.shape[0])

    result = (
        total_displacement,
        mean_of_feature_per_fps(speed_per_frame),
        mean_of_feature_per_fps(acceleration_per_frame),
    )
    return np.array(result) if as_array else result
