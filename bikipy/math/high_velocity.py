import numpy as np
from numba import njit
from pydantic_numpy.typing import Np2DArrayFp64

from bikipy import runtime_settings


def high_velocity_removal(position_array: Np2DArrayFp64, max_distance_per_frame: float) -> Np2DArrayFp64:
    """
    We have an array of coordinates, referred to as 'position_array', representing positions in each frame. Our goal is
    to identify and remove points where the movement speed between frames exceeds a certain limit,
    defined as 'max_distance_per_frame'.

    First, we calculate the velocity between each pair of consecutive points using the np.diff function.
    This gives us an array of velocities.

    Next, we compare these velocities with our maximum allowed speed. We use a logical operation
    (greater than or equal to, GE) to find out which velocities are too high, exceeding our threshold.
    This comparison results in an array of True/False values, indicating whether each velocity is above the threshold.

    We then locate the indices of these excessive velocities in our original position array. For each of these indices
    (let's call it 'high_v_idx'), we examine the surrounding points to decide if they should be removed.
    Specifically, we check the velocity between the point at 'high_v_idx' and the point two steps ahead (at 'high_v_idx+2').
    If this velocity is below our threshold, we conclude that only the point at 'high_v_idx+1' caused the high velocity,
    and we replace its coordinates with 'np.nan' to remove it.

    This check continues for subsequent points (high_v_idx+3, high_v_idx+4, etc.)
    until we either find a point that doesn't cause high velocity or reach the end of the array.
    The aim is to only remove the points that are responsible for the high velocity,
    while keeping the rest of the data intact.

    :param position_array:
    :param max_distance_per_frame:
    :return:
    """
    velocities = np.abs(np.diff(np.linalg.norm(position_array, axis=1), axis=0))

    high_velocity_indices = np.where(velocities >= max_distance_per_frame)[0]

    for idx in high_velocity_indices:
        # Skip if the index has already been evaluated and set to np.nan
        if np.isnan(position_array[idx]).any():
            continue

        position_array[idx + 1] = np.nan

        check_idx = idx + 2
        # Continue checking until a point is found with velocity below the threshold or the end of the array is reached
        current_max_distance = max_distance_per_frame
        while check_idx < len(position_array):
            velocity = np.linalg.norm(position_array[idx] - position_array[check_idx])
            if velocity < current_max_distance:
                break

            # Set the current checking point to np.nan and move to the next point
            position_array[check_idx] = np.nan

            check_idx += 1
            current_max_distance += max_distance_per_frame

    return position_array


if not runtime_settings.disable_numba:
    high_velocity_removal = njit(cache=True)(high_velocity_removal)
