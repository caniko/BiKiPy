from typing import Sequence, SupportsFloat

import numpy as np


def _get_mean_feature(feature: np.ndarray, fps: SupportsFloat):
    seconds = feature.shape[0] / float(fps)
    return np.sum(feature) / seconds


def displacement(location: Sequence) -> np.ndarray:
    return np.sum(np.linalg.norm(np.diff(location, axis=0), axis=1))


def mean_speed(location: Sequence, fps: SupportsFloat) -> np.ndarray:
    location = np.asanyarray(location)
    total_speed = np.linalg.norm(np.diff(np.diff(location, axis=0), axis=0), axis=1)

    return _get_mean_feature(total_speed, fps)


def mean_acceleration(location: Sequence, fps: SupportsFloat) -> np.ndarray:
    location = np.asanyarray(location)
    total_acceleration = np.linalg.norm(
        np.diff(np.diff(np.diff(location, axis=0), axis=0), axis=1), axis=1
    )

    return _get_mean_feature(total_acceleration, fps)


def displacement_mean_speed_acceleration(location: Sequence, fps: SupportsFloat):
    location = np.asanyarray(location)

    displacement_per_frame = np.diff(location, axis=0)
    speed_per_frame = np.diff(displacement_per_frame, axis=0)
    acceleration_per_frame = np.diff(displacement_per_frame, axis=0)

    total_speed = np.linalg.norm(speed_per_frame, axis=1)
    total_acceleration = np.linalg.norm(acceleration_per_frame, axis=1)

    return (
        np.sum(np.linalg.norm(np.diff(displacement_per_frame, axis=0), axis=1)),
        _get_mean_feature(total_speed, fps),
        _get_mean_feature(total_acceleration, fps),
    )
