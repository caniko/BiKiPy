from functools import cached_property
from logging import getLogger
from typing import Any, Iterable

import numpy as np
import pandas as pd

from bikipy.core.base_class import BikipyBase
from bikipy.core.typing import NDArrayBool, NDArrayFp64
from bikipy.utils.math.calculus import np_abs_diff
from bikipy.utils.misc import generic_multi_indexer

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


def total_displacement_median_speed_acceleration(
    coordinate_sequence: NDArrayFp64,
    meters_per_pixel: NDArrayFp64 | float,
    fps: float,
) -> tuple:
    """

    Parameters
    ----------
    coordinate_sequence
    meters_per_pixel
    fps

    Returns
    -------
    (total displacement, speed per frame, acceleration per frame)
    """
    displacement = (
        displacement_by_frame(coordinate_sequence) * meters_per_pixel
        if isinstance(meters_per_pixel, float)
        else displacement_by_frame(coordinate_sequence * np.asarray(meters_per_pixel))
    )
    if np.any(displacement):
        return (
            np.sum(displacement),
            np.nanmedian((speed := np_abs_diff(displacement) * fps)),
            np.nanmedian(np_abs_diff(speed)),
        )
    else:
        return 0, 0, 0


def frozen_frames(
    fps: float,
    rigid_body_node_displacements: Iterable[NDArrayFp64],
    second_threshold: float = 1.0,
    metric_displacement_threshold: float = 0.005,
) -> NDArrayFp64:
    """
    Compute the time the rigid body has been frozen or "stood still" throughout
    the trial. The acceleration at these frames should be close to zero.

    Formal definition: Given that the body is immobile up to a certain tolerance,
    defined by metric_displacement_threshold, for longer than the defined second threshold
    define the respective sequence as frozen. True = Frozen; False = Mobile

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
    def metric_displacement_by_frame(self):
        return displacement_by_frame(self.coordinate_sequence * self.meters_per_pixel)

    @cached_property
    def total_displacement(self):
        return np.nansum(self.metric_displacement_by_frame)

    @cached_property
    def speed(self):
        if not self.total_displacement:
            return np.nan
        return np_abs_diff(self.metric_displacement_by_frame) * self.fps

    @cached_property
    def median_speed(self):
        if not self.total_displacement:
            return np.nan
        return np.nanmedian(self.speed)

    @cached_property
    def frozen_boolean_index(self) -> NDArrayBool:
        if not self.total_displacement:
            return np.nan
        return frozen_frames(self.fps, (self.metric_displacement_by_frame,))

    @cached_property
    def freezing_time(self):
        if not self.total_displacement:
            return np.nan
        return np.nansum(self.frozen_boolean_index) / self.fps

    @cached_property
    def acceleration(self):
        if not self.total_displacement:
            return np.nan
        return np_abs_diff(self.speed)

    @cached_property
    def median_acceleration(self):
        if not self.total_displacement:
            return np.nan
        return np.nanmedian(self.acceleration)

    @property
    def to_list(self):
        return [
            self.total_displacement,
            self.median_speed,
            self.median_acceleration,
            self.freezing_time,
        ]


def motion_multi_indexer(category: Any, level: int):
    return generic_multi_indexer(
        "Displacement", "Median_speed", "Median_speed", "Median_acceleration", "Freezing time"
    )(category, level)


def get_combined_features_from_merged_motion_island_data(
    boolean_index: NDArrayBool,
    coordinate_sequence: NDArrayFp64,
    meters_per_pixel,
    fps: float,
    minimum_seconds_of_data: float = 4.0,
):
    def motion_object_from_slice(slice_start, slice_end) -> list:
        return Motion(
            coordinate_sequence=coordinate_sequence[slice_start:slice_end],
            meters_per_pixel=meters_per_pixel,
            fps=fps,
        ).to_list

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

    i, start, end = find_index_start_n_end()
    motion_features = []
    while i < number_of_frames:
        potential_end = indexes[i]
        next_step_from_end = end + 1
        if potential_end == next_step_from_end:
            end = potential_end
        elif potential_end < next_step_from_end:
            if potential_end - end > minimum_frames:
                motion_features.append(motion_object_from_slice(start, end))
                i, start, end = find_index_start_n_end(i)
                continue
            else:
                end = potential_end
        i += 1

    if (end := indexes[-1] + 1) - start >= 4:
        motion_features.append(motion_object_from_slice(start, end))

    if not motion_features or not np.any(motion_features[0]):
        return _zero_return

    return {
        "total_displacement": sum(motion_feature[0] for motion_feature in motion_features),
        "median_speed": np.nanmean([motion_feature[1] for motion_feature in motion_features]),
        "median_acceleration": np.nanmean([motion_feature[2] for motion_feature in motion_features]),
        "freezing_time": sum(motion_feature[3] for motion_feature in motion_features),
    }
