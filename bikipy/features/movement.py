from typing import Sequence, SupportsFloat

import numpy as np


def _mean_of_feature_per_fps(feature: np.ndarray, fps: SupportsFloat):
    feature = np.asanyarray(feature)
    return np.sum(feature) / float(fps)


def displacement(location_sequence: Sequence[Sequence[SupportsFloat]]) -> np.ndarray:
    location_sequence = np.asanyarray(location_sequence)
    return np.sum(np.linalg.norm(np.diff(location_sequence, axis=0), axis=1))


def speed(location_sequence: Sequence[Sequence[SupportsFloat]]) -> np.ndarray:
    location_sequence = np.asanyarray(location_sequence)
    return np.diff(displacement(location_sequence), axis=0)


def acceleration(location_sequence: Sequence[Sequence[SupportsFloat]]) -> np.ndarray:
    location_sequence = np.asanyarray(location_sequence)
    return np.diff(speed(location_sequence), axis=0)


def mean_speed(
    location_sequence: Sequence[Sequence[SupportsFloat]], fps: SupportsFloat
) -> np.ndarray:
    location_sequence = np.asanyarray(location_sequence)
    return _mean_of_feature_per_fps(speed(location_sequence), fps)


def mean_acceleration(
    location_sequence: Sequence[Sequence[SupportsFloat]], fps: SupportsFloat
) -> np.ndarray:
    location_sequence = np.asanyarray(location_sequence)
    return _mean_of_feature_per_fps(acceleration(location_sequence), fps)


def displacement_mean_speed_acceleration(
    location_sequence: Sequence[Sequence[SupportsFloat]],
    fps: SupportsFloat,
    as_array: bool = True,
):
    location_sequence = np.asanyarray(location_sequence)

    displacement_per_frame = np.diff(location_sequence, axis=0)
    speed_per_frame = np.diff(displacement_per_frame, axis=0)
    acceleration_per_frame = np.diff(speed_per_frame, axis=0)

    total_speed = np.linalg.norm(speed_per_frame, axis=1)
    total_acceleration = np.linalg.norm(acceleration_per_frame, axis=1)

    result = (
        np.sum(np.linalg.norm(np.diff(displacement_per_frame, axis=0), axis=1)),
        _mean_of_feature_per_fps(total_speed, fps),
        _mean_of_feature_per_fps(total_acceleration, fps),
    )
    return np.array(result) if as_array else result
