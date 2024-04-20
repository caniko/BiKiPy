from functools import cached_property, lru_cache
from logging import getLogger
from typing import Iterable, Optional

import numpy as np
import pandas as pd
from pydantic import Field, computed_field, validate_call
from pydantic_numpy.typing import Np1DArrayBool, NpNDArrayFp64

from bikipy.core.base import BikipyModel
from bikipy.math.calculus import np_abs_diff
from bikipy.math.discrete import TruthIslandMetadata
from bikipy.math.statistics import nan_average
from bikipy.utils.pandas import generic_multi_indexer

logger = getLogger(__name__)

summary_motion_features = (
    "total_displacement",
    "median_speed",
    "median_acceleration",
    "freezing_time",
)
_zero_return = {feature: 0.0 for feature in summary_motion_features}


def displacement_by_frame(
    coordinate_sequence: NpNDArrayFp64,
    interpolation_method: str = "akima",
    remove_tails: bool = False,
) -> NpNDArrayFp64:
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
    rigid_body_node_displacements: Iterable[NpNDArrayFp64],
    second_threshold: float = 1.0,
    metric_displacement_threshold: float = 0.005,
) -> NpNDArrayFp64:
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
    :type rigid_body_node_displacements: NpNDArrayFp64
    :type second_threshold: float
    :type metric_displacement_threshold: float
    :return: Boolean index storing the freezing state of the animal across frames
    :rtype: NpNDArrayFp64
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


class Motion(BikipyModel):
    coordinate_sequence: NpNDArrayFp64
    timestamp_sequence: Optional[NpNDArrayFp64] = None
    fps: Optional[float] = None
    weight: Optional[int] = Field(
        None, description="The weight of the Motion instance defines relative weight to related Motion instances"
    )

    @computed_field  # type: ignore[misc]
    @cached_property
    def int_fps(self) -> int:
        return round(self.fps)

    @computed_field  # type: ignore[misc]
    @cached_property
    def meters_per_frame(self) -> NpNDArrayFp64:
        return displacement_by_frame(self.coordinate_sequence)

    @computed_field  # type: ignore[misc]
    @cached_property
    def meters_per_second(self) -> NpNDArrayFp64:
        if self.timestamp_sequence is not None:
            return [
                np.nansum(self.meters_per_frame[i : i + self.int_fps])
                for i in range(0, self.meters_per_frame.size, self.int_fps)
            ]
        return [
            np.nansum(self.meters_per_frame[i : i + self.int_fps])
            for i in range(0, self.meters_per_frame.size, self.int_fps)
        ]

    @computed_field  # type: ignore[misc]
    @cached_property
    def total_displacement(self) -> float:
        return np.nansum(self.meters_per_frame)

    @computed_field  # type: ignore[misc]
    @cached_property
    def median_speed(self) -> float:
        if not self.total_displacement:
            return np.nan
        return np.nanmedian(self.meters_per_second)

    @computed_field  # type: ignore[misc]
    @cached_property
    def frozen_boolean_index(self) -> Np1DArrayBool:
        if not self.total_displacement:
            return np.nan
        return frozen_frames(self.fps, (self.meters_per_frame,))

    @computed_field  # type: ignore[misc]
    @cached_property
    def freezing_time(self) -> float:
        if not self.total_displacement:
            return np.nan
        return np.nansum(self.frozen_boolean_index) / self.fps

    @computed_field  # type: ignore[misc]
    @cached_property
    def acceleration(self) -> NpNDArrayFp64:
        if not self.total_displacement:
            return np.nan
        return np_abs_diff(self.coordinate_sequence)

    @computed_field  # type: ignore[misc]
    @cached_property
    def median_acceleration(self) -> float:
        if not self.total_displacement:
            return np.nan
        return np.nanmedian(self.acceleration)

    @computed_field  # type: ignore[misc]
    @property
    def as_tuple(self) -> tuple:
        if self.weight:
            return self.total_displacement, self.median_speed, self.median_acceleration, self.freezing_time, self.weight
        return self.total_displacement, self.median_speed, self.median_acceleration, self.freezing_time

    def lowpass_std_filter(self, sd_scale: float = 2.0):
        displacement = np.diff(self.coordinate_sequence, axis=0)
        displacement_sd = np.std(np.abs(displacement), axis=0)

        filter_boolean_idx = displacement > (displacement_sd * sd_scale)
        if not np.any(filter_boolean_idx):
            return []

        return np.where(filter_boolean_idx)[0]


EMPTY_MOTION = np.full(4, np.nan)
EMPTY_MOTION_WEIGHT = np.full(5, np.nan)


@lru_cache
@validate_call
def motion_analysis_indexer(category: str, level: int):
    assert level >= 2, "Must be at least 2 levels"
    return generic_multi_indexer("Displacement", "MedianSpeed", "MedianAcceleration", "FreezingTime")(category, level)


@lru_cache
@validate_call
def bulk_motion_analysis_indexer(categories: tuple[str, ...], level: int):
    result = []
    for category in categories:
        result.extend(motion_analysis_indexer(category, level))
    return result


def merge_motion_island_data(motion_islands: TruthIslandMetadata, coordinate_sequence: NpNDArrayFp64, fps: float):
    """
    The purpose of this function is to deal with islands of data that need to be aggregated for analysis. These islands
    of data have to be merged arbitrarily.

    A simple merge would make the computation of speed and acceleration wrong.
    """
    if not motion_islands:
        return _zero_return

    df = pd.DataFrame(
        [
            Motion(coordinate_sequence=coordinate_sequence[start:end], fps=fps, weight=length).as_tuple
            for start, end, length in motion_islands
        ],
        columns=["total_displacement", "median_speed", "median_acceleration", "freezing_time", "weight"],
    )

    # Making sure to not have any np.nans before np.average
    df.dropna(axis=0, inplace=True)

    return {
        "total_displacement": df["total_displacement"].sum(),
        "median_speed": 0.0 if np.all(np.isnan(df["median_speed"])) else nan_average(df["median_speed"], df["weight"]),
        "median_acceleration": (
            0.0 if np.all(np.isnan(df["median_acceleration"])) else nan_average(df["median_acceleration"], df["weight"])
        ),
        "freezing_time": df["freezing_time"].sum(),
    }
