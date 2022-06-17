from functools import cached_property
from logging import getLogger
from typing import Any, Iterable

import numpy as np
import pandas as pd

from bikipy.core.base_class import BikipyBase
from bikipy.core.typing import NDArrayBool, NDArrayFp64
from bikipy.utils.collection_utils import generic_multi_indexer
from bikipy.utils.math.calculus import np_abs_diff

logger = getLogger(__name__)


summary_motion_features = (
    "total_displacement",
    "median_speed",
    "median_acceleration",
    "freezing_time",
)
_zero_return = {feature: 0.0 for feature in summary_motion_features}


def units_pixels_per_second_frame(meters_per_pixel: float, fps: float):
    return meters_per_pixel * fps


def displacement_by_frame(
    coordinate_sequence: NDArrayFp64,
    interpolation_method: str = "akima",
    remove_tails: bool = False,
) -> NDArrayFp64:
    """
    Compute the absolute displacement of the given point from its coordinates across frames.
    The values on the tails are removed if they are undefined or "not a number" (NaN). The
    undefined values are interpolated.
    """
    if np.all(np.isnan((magnitudes := np.linalg.norm(coordinate_sequence, axis=1)))):
        return np_abs_diff(magnitudes)
    logger.debug("Interpolating data as there are non-finite values in the location data")

    magnitudes_series = pd.Series(magnitudes)
    magnitudes_series.interpolate(
        method=interpolation_method,
        limit_direction="both",
        limit_area="inside" if remove_tails else None,
        inplace=True,
    )
    if remove_tails:
        magnitudes_series.dropna(inplace=True)

    return np_abs_diff(magnitudes_series.values)


def frozen_frames(
    fps: float,
    rigid_body_node_displacements: Iterable[NDArrayFp64],
    second_threshold: float = 1.0,
    metric_displacement_threshold: float = 0.005,
) -> NDArrayFp64:
    """
    Compute the time the rigid body has been frozen or "stood still" throughout
    the trial. The acceleration at these frames should be close to zero.

    Formal definition:
    1. The body is immobile within the defined upper boundary, metric_displacement_threshold
    2. For a longer time than the defined second threshold, the sequence is defined as frozen.

    True = Frozen; False = Mobile

    Method:
        #. Filter each node in the rigid body discretely with both thresholds
        #. Perform logical AND operation on the result from the nodes

        Some of the freeze epochs may be orphaned as they are below the time threshold
        after the AND operation, which is why a last step is necessary:

        #. Filter the result from 2. with respect to the time/frame/second threshold

    :param fps: Frames per second (fps) of the video the data was collected from
    :param rigid_body_node_displacements:
    :param second_threshold:
    :param metric_displacement_threshold:
    :type fps: float
    :type rigid_body_node_displacements: NDArrayFp64
    :type second_threshold: float
    :type metric_displacement_threshold: float
    :return: Boolean index storing the freezing state of the animal across frames
    :rtype: NDArrayFp64
    """
    frame_threshold = round(second_threshold * fps)

    discrete_thresholding = []
    for displacement in rigid_body_node_displacements:
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

    logical_and_thresholding = np.logical_and.reduce(discrete_thresholding)

    start = 0
    while start < logical_and_thresholding.size:
        if logical_and_thresholding[start]:
            end = start + 1
            while end < logical_and_thresholding.size and logical_and_thresholding[end]:
                end += 1

            if end - start < frame_threshold:
                logical_and_thresholding[start : end + 1] = False
            if end < logical_and_thresholding.size:
                break
            start = end + 1
        else:
            start += 1

    return logical_and_thresholding


class Motion(BikipyBase):
    coordinate_sequence: NDArrayFp64
    meters_per_pixel: float | NDArrayFp64
    fps: float

    @cached_property
    def int_fps(self) -> int:
        return round(self.fps)

    @cached_property
    def meters_per_frame(self) -> NDArrayFp64:
        return displacement_by_frame(self.coordinate_sequence * self.meters_per_pixel)

    @cached_property
    def meters_per_second(self) -> NDArrayFp64:
        return [
            np.nansum(self.meters_per_frame[i : i + self.int_fps])
            for i in range(0, self.meters_per_frame.size, self.int_fps)
        ]

    @cached_property
    def total_displacement(self) -> float:
        return np.nansum(self.meters_per_frame)

    @cached_property
    def median_speed(self) -> float:
        if not self.total_displacement:
            return np.nan
        return np.nanmedian(self.meters_per_second)

    @cached_property
    def frozen_boolean_index(self) -> NDArrayBool:
        if not self.total_displacement:
            return np.nan
        return frozen_frames(self.fps, (self.meters_per_frame,))

    @cached_property
    def freezing_time(self) -> float:
        if not self.total_displacement:
            return np.nan
        return np.nansum(self.frozen_boolean_index) / self.fps

    @cached_property
    def acceleration(self) -> NDArrayFp64:
        if not self.total_displacement:
            return np.nan
        return np_abs_diff(self.meters_per_second)

    @cached_property
    def median_acceleration(self) -> float:
        if not self.total_displacement:
            return np.nan
        return np.nanmedian(self.acceleration)

    @property
    def as_tuple(self) -> tuple[float, float, float, float]:
        return self.total_displacement, self.median_speed, self.median_acceleration, self.freezing_time


def motion_multi_indexer(category: Any, level: int):
    return generic_multi_indexer("Displacement", "MedianSpeed", "MedianAcceleration", "FreezingTime")(category, level)


def get_combined_features_from_merged_motion_island_data(
    boolean_index: NDArrayBool,
    coordinate_sequence: NDArrayFp64,
    meters_per_pixel,
    fps: float,
    minimum_seconds_of_data: float = 4.0,
):
    def motion_object_from_slice(slice_start, slice_end) -> Motion:
        return Motion(
            coordinate_sequence=coordinate_sequence[slice_start:slice_end],
            meters_per_pixel=meters_per_pixel,
            fps=fps,
        )

    def find_index_start_n_end(starting_index: int = 0):
        new_start, new_end = indexes[starting_index], indexes[starting_index := starting_index + 1]
        while new_end - new_start > fps:
            new_start, new_end = indexes[starting_index], indexes[starting_index := starting_index + 1]
        return starting_index + 1, new_start, new_end

    number_of_frames = np.sum(boolean_index)
    minimum_frames = minimum_seconds_of_data * fps
    if number_of_frames < minimum_frames:
        return _zero_return

    indexes = np.where(boolean_index)[0]
    third_of_a_second = fps / 3.0

    last_index = number_of_frames - 1
    i, start, _end = find_index_start_n_end()

    data = []
    while i < number_of_frames:
        potential_end = indexes[i]
        next_step_from_previous_end = indexes[i - 1] + 1
        if potential_end == next_step_from_previous_end:
            pass
        elif potential_end > next_step_from_previous_end:
            jump_length = potential_end - next_step_from_previous_end
            if jump_length <= third_of_a_second:
                end = potential_end

                # Look ahead before committing to end index
                if i != last_index and end - indexes[i + 1] < third_of_a_second:
                    i += 1
                    continue

            else:
                end = indexes[i - 1]

            if (slice_len := end - start) > minimum_frames:
                motion = motion_object_from_slice(start, end)
                if not np.isnan(motion.median_acceleration):
                    data.append((*motion.as_tuple, slice_len))

                if i == last_index:
                    break

                i, start, end = find_index_start_n_end(i)
                continue
        else:  # potential_end < next_step_from_previous_end
            msg = "potential_end < next_step_from_end cannot be true in a sorted index"
            raise RuntimeError(msg)

        i += 1

    if not data:
        return _zero_return

    df = pd.DataFrame(
        data, columns=["total_displacement", "median_speed", "median_acceleration", "freezing_time", "weight"]
    )
    # Making sure to not have any np.nans before np.average
    df.dropna(axis=0, how="any", thresh=None, subset=None, inplace=True)

    return {
        "total_displacement": df["total_displacement"].sum(),
        "median_speed": np.average(df["median_speed"], weights=df["weight"]),
        "median_acceleration": np.average(df["median_acceleration"], weights=df["weight"]),
        "freezing_time": df["freezing_time"].sum(),
    }
