from collections.abc import Sequence
from logging import getLogger
from typing import Union, Iterable

import numpy as np
import pandas as pd

from bikipy.math.calculus import absolute_derivative

logger = getLogger(__name__)


def units_pixels_per_second_frame(
    units_per_pixel: Union[float, int], fps: Union[float, int]
):
    return units_per_pixel * fps


def displacement_per_frame(
    coordinate_sequence: Sequence[Sequence[float]],
    interpolation_method: str = "akima",
    remove_tails: bool = False,
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
    remove_tails

    Returns
    -------
    np.ndarray with pixel displacement per frame
    """
    if np.all(np.isnan((magnitudes := np.linalg.norm(coordinate_sequence, axis=1)))):
        return absolute_derivative(magnitudes)

    logger.debug(
        "Interpolating data as there are non-finite values in the location data"
    )

    magnitudes_series = pd.Series(magnitudes)
    magnitudes_series.interpolate(
        method=interpolation_method,
        limit_direction="both",
        limit_area="inside" if remove_tails else None,
        inplace=True,
    )
    if remove_tails:
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
            self.speed = absolute_derivative(self.metric_displacement_per_frame) * fps
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


def freezing_time(
    fps: Union[float, int],
    displacements: Iterable[np.ndarray],
    second_threshold: float = 1.0,
    metric_displacement_threshold: float = 0.005,
):
    """
    Compute the time the rigid body has been frozen or "stood still" throughout
    the trial

    Formal definition: Given that the body is immobile up to a certain tolerance,
    defined by metric_displacement_threshold, for longer than second_threshold

    Method:
        1. Filter each node in the rigid body discretely with both thresholds
        2. Perform logical AND operation on the result from the nodes

        Some of the freeze epochs may be orphaned as they below the time threshold
        after the AND operation, which is why a last step is necessary.

        3. Filter the result from 2. with respect to the time/frame/second threshold

    Parameters
    ----------
    fps
    displacements
    second_threshold
    metric_displacement_threshold

    Returns
    -------

    """
    frame_threshold = round(second_threshold * fps)

    discrete_thresholding = []
    for displacement in displacements:
        displacement = np.asarray(displacement)
        result = np.zeros(displacement.shape[0], dtype=bool)

        start, end = 0, frame_threshold
        while end < result.size:
            """
            Do-while loop-like; stops when end is larger than result length.
            frame_threshold is utilized implicitly; the difference between
            end and start can never be lower than the frame_threshold
            """
            range_sum = np.sum(displacement[start:end])
            if range_sum <= metric_displacement_threshold:
                while end < result.size:
                    range_sum += displacement[end]
                    end += 1

                    if range_sum > metric_displacement_threshold:
                        result[start:end] = True
                        break

                start = end
                end += frame_threshold
            else:
                start += 1
                end += 1

        discrete_thresholding.append(result)

    logical_and_thresholding = np.logical_and.reduce(np.array(discrete_thresholding))

    start = 0
    while start < logical_and_thresholding.size:
        if logical_and_thresholding[start]:
            end = start + 1
            while logical_and_thresholding[end] and end < logical_and_thresholding.size:
                end += 1

            if end - start < frame_threshold:
                logical_and_thresholding[start : end + 1] = False
            if end < logical_and_thresholding.size:
                break
            start = end + 1
        else:
            start += 1

    return logical_and_thresholding


def attention_per_frame(
    valid_indices: Sequence[bool],
    fps: float,
    minimum_seconds_observing: float = 0.2,
) -> np.ndarray:
    """
    Compute the attentiveness of the boolean array.

    The boolean array should reflect instances of observation in binary, True or False.



    :param valid_indices:
    :param fps:
    :param minimum_seconds_observing:
    :return:
    """
    valid_indices = np.asarray(valid_indices)

    fps = float(fps)
    minimum_seconds_observing = float(minimum_seconds_observing)

    distraction_tolerance = round(fps / 2.0)
    minimum_time_valid_observation = round(minimum_seconds_observing * fps)

    length = valid_indices.shape[0]
    observation_boolean_indices = np.zeros(length, dtype=bool)

    first_valid_index = None
    i, valid_frames_within_border, true_counter, consecutive_false = 0, 0, 0, 0
    while True:
        if valid_indices[i]:
            true_counter += 1

            if consecutive_false:
                true_counter += consecutive_false - 1  # make up for the previous += 1
                consecutive_false = 0

            if true_counter == minimum_time_valid_observation:
                first_valid_index = i - minimum_time_valid_observation + 1

        else:
            if first_valid_index is not None:
                if consecutive_false <= distraction_tolerance:
                    consecutive_false += 1
                else:
                    observation_boolean_indices[first_valid_index : i + 1] = True
                    valid_frames_within_border += true_counter

                    consecutive_false, true_counter = 0, 0
                    first_valid_index = None

            else:
                true_counter = 0

        i += 1

        if i == length:
            if first_valid_index is not None:
                observation_boolean_indices[first_valid_index:] = True
                valid_frames_within_border += true_counter
            break

    if valid_frames_within_border == 0:
        logger.info(f"Subject didn't observe the nort object")

        assert not np.any(observation_boolean_indices)
        return observation_boolean_indices

    assert (
        np.any(observation_boolean_indices)
        and np.sum(observation_boolean_indices) >= minimum_time_valid_observation
    ), (
        f"True: {np.sum(observation_boolean_indices)}; fps: {fps}; "
        f"Minimum observation frames: {minimum_time_valid_observation}"
    )

    return observation_boolean_indices
