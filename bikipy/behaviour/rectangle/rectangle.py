import os
from abc import ABC
from functools import cached_property, lru_cache
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
from numpy.typing import NDArray
from pydantic import validate_arguments, validator
from skg import ngauss_fit

from bikipy.behaviour.base import BaseExperiment, BaseTrial
from bikipy.behaviour.mixin.meters_per_pixel.resolution_derived import (
    ResolutionDerivedUnitPerPixelMixin,
    ResolutionDerivedUnitPerPixelTrialMixin,
)
from bikipy.behaviour.utils import reduce_repeating_sequences
from bikipy.core.base_class import BikipyBase
from bikipy.feature.motion import (
    get_combined_features_from_merged_motion_island_data,
    motion_2d_multi_indexer,
)
from bikipy.math.point_in_polygon import parallel_point_in_polygon

A = 255
QUADRANT_INSPECTION_DIR_NAME = "PiP_quadrant_location_booleans"
CENTER_INSPECTION_DIR_NAME = "PiP_center_location_booleans"


class Quadrant(BikipyBase):
    corners: NDArray
    coordinates_per_frame: NDArray
    meters_per_pixel: float
    fps: float

    @validator("corners")
    def make_contiguous_array(cls, value):
        return np.ascontiguousarray(value, dtype=np.float32)

    @cached_property
    def confinement_boolean_index(self):
        return parallel_point_in_polygon(self.coordinates_per_frame, self.corners)

    @cached_property
    def motion(self) -> dict:
        return get_combined_features_from_merged_motion_island_data(
            self.confinement_boolean_index,
            self.coordinates_per_frame,
            self.meters_per_pixel,
            self.fps,
        )


class RectangleEnclosedExperiment(BaseExperiment, ResolutionDerivedUnitPerPixelMixin):
    @cached_property
    def motion_summary_columns(self) -> list:
        quadrant_labels = (
            "Upper-left Quadrant",
            "Upper-right Quadrant",
            "Lower-left Quadrant",
            "Lower-right Quadrant",
        )
        return super().motion_summary_columns + [
            ("gaussian_center_to_periphery_score", ""),
            *motion_2d_multi_indexer("Upper-left Quadrant"),
            *motion_2d_multi_indexer("Upper-right Quadrant"),
            *motion_2d_multi_indexer("Lower-left Quadrant"),
            *motion_2d_multi_indexer("Lower-right Quadrant"),
            *self._feature_2d_multi_indexer("Seconds present", quadrant_labels),
            *self._feature_2d_multi_indexer("Entries", quadrant_labels),
        ]

    def _make_categorical_inspection_dir(self, trial_root_dir: Path):
        os.makedirs(trial_root_dir / "")


class RectangleEnclosedTrial(BaseTrial, ResolutionDerivedUnitPerPixelTrialMixin, ABC):
    center_box_to_recording_resolution_ratio: Optional[float] = None
    rectangle_2d_bin: tuple[int, int] = (2, 2)

    @cached_property
    def _quadrant_inspection_dir(self):
        return self.inspection_dir / QUADRANT_INSPECTION_DIR_NAME

    @validator("inspection_dir")
    def make_categorical_inspection_sub_dirs(cls, value):
        if (
            value
            and not (quadrant_dir := value / QUADRANT_INSPECTION_DIR_NAME).exists()
        ):
            os.mkdir(quadrant_dir)
            for current_quadrant in (
                "upper_left",
                "upper_right",
                "lower_right",
                "lower_left",
            ):
                os.mkdir(quadrant_dir / current_quadrant)
            os.mkdir(value / CENTER_INSPECTION_DIR_NAME)
        return value

    @cached_property
    def gaussian_center_to_periphery_score(self):
        func = gaussian_scoring_field(self.tuple_recording_resolution)
        scores = np.array(
            [func(*coordinate) for coordinate in self.coordinates_per_frame]
        )
        return np.sum(scores) / (A * self.number_of_frames)

    @cached_property
    def quadrant_corners(self) -> dict[tuple[int, int], Quadrant]:
        """
        Left to right, top to down
        :return:
        """
        horizontal_uniform_distance = (
            self.horizontal_resolution / self.rectangle_2d_bin[0]
        )
        vertical_uniform_distance = self.vertical_resolution / self.rectangle_2d_bin[1]
        result = {}
        for v in range(1, self.rectangle_2d_bin[1]):
            vertical_coordinate_min = vertical_uniform_distance * (v - 1)
            vertical_coordinate_max = vertical_uniform_distance * v
            for h in range(1, self.rectangle_2d_bin[0]):
                result[(h, v)] = (h * horizontal_uniform_distance, vertical_coordinate)
        return

    @cached_property
    def location_sequence_quadrant(self) -> np.ndarray:
        result = self._uint_zeros_based_on_frame_length.copy()

        result[self.quadrant_upper_left_boolean_index] = 1
        result[self.quadrant_upper_right_boolean_index] = 2
        result[self.quadrant_lower_left_boolean_index] = 3
        result[self.quadrant_lower_right_boolean_index] = 4

        return reduce_repeating_sequences(result, round(self.fps * 0.35))

    # Quadrant upper left 1
    @cached_property
    def quadrant_upper_left_boolean_index(self) -> np.ndarray:
        return parallel_point_in_polygon(
            np.array((0.0, 0.0)),
            np.array((self.recording_center_pixel[0], 0.0)),
            np.array((0.0, self.recording_center_pixel[1])),
            self.coordinates_per_frame,
            inspect=self._quadrant_inspection_dir
            / "upper_left"
            / self._inspection_image_name
            if self.inspection_dir
            else None,
            inspect_image=self.inspect_image,
        )

    @cached_property
    def quadrant_upper_left_entries(self) -> int:
        return np.sum(self.location_sequence_quadrant == 1)

    @cached_property
    def seconds_on_quadrant_upper_left(self):
        return np.sum(self.quadrant_upper_left_boolean_index) / self.fps

    # Quadrant upper right 2

    @cached_property
    def quadrant_upper_right_boolean_index(self) -> np.ndarray:
        return parallel_point_in_polygon(
            np.array((self.recording_center_pixel[0], 0.0)),
            self.recording_center_pixel,
            np.array((self.horizontal_resolution, 0.0)),
            self.coordinates_per_frame,
            inspect=self._quadrant_inspection_dir
            / "upper_right"
            / self._inspection_image_name
            if self.inspection_dir
            else None,
            inspect_image=self.inspect_image,
        )

    @cached_property
    def quadrant_upper_right_entries(self):
        return np.sum(self.location_sequence_quadrant == 2)

    @cached_property
    def seconds_on_quadrant_upper_right(self):
        return np.sum(self.quadrant_upper_right_boolean_index) / self.fps

    # Quadrant lower left 3

    @cached_property
    def quadrant_lower_left_boolean_index(self) -> np.ndarray:
        return parallel_point_in_polygon(
            np.array((0.0, self.vertical_resolution)),
            np.array((self.recording_center_pixel[0], self.vertical_resolution)),
            np.array((0.0, self.recording_center_pixel[1])),
            self.coordinates_per_frame,
            inspect=self._quadrant_inspection_dir
            / "lower_left"
            / self._inspection_image_name
            if self.inspection_dir
            else None,
            inspect_image=self.inspect_image,
        )

    @cached_property
    def quadrant_lower_left_entries(self):
        return np.sum(self.location_sequence_quadrant == 3)

    @cached_property
    def seconds_on_quadrant_lower_left(self):
        return np.sum(self.quadrant_lower_left_boolean_index) / self.fps

    # Quadrant lower right 4

    @cached_property
    def quadrant_lower_right_boolean_index(self) -> np.ndarray:
        return parallel_point_in_polygon(
            np.array((self.recording_center_pixel[0], self.vertical_resolution)),
            self.recording_resolution,
            self.recording_center_pixel,
            self.coordinates_per_frame,
            inspect=self._quadrant_inspection_dir
            / "lower_right"
            / self._inspection_image_name
            if self.inspection_dir
            else None,
            inspect_image=self.inspect_image,
        )

    @cached_property
    def quadrant_lower_right_entries(self):
        return np.sum(self.location_sequence_quadrant == 4)

    @cached_property
    def seconds_on_quadrant_lower_right(self):
        return np.sum(self.quadrant_lower_right_boolean_index) / self.fps

    @property
    def motion_features(self):
        return super().motion_features + [
            self.gaussian_center_to_periphery_score,
            *self.motion_quadrant_upper_left.values(),
            *self.motion_quadrant_upper_right.values(),
            *self.motion_quadrant_lower_left.values(),
            *self.motion_quadrant_lower_right.values(),
            self.seconds_on_quadrant_upper_left,
            self.seconds_on_quadrant_upper_right,
            self.seconds_on_quadrant_lower_left,
            self.seconds_on_quadrant_lower_right,
            self.quadrant_upper_left_entries,
            self.quadrant_upper_right_entries,
            self.quadrant_lower_left_entries,
            self.quadrant_lower_right_entries,
        ]

    # Center vs Periphery ==============================================================
    @cached_property
    def center_square_corners(self):
        if self.center_box_to_recording_resolution_ratio is None:
            msg = "center_box_to_recording_resolution_ratio must be defined for center and periphery analysis"
            raise AttributeError(msg)
        center_pixel_lengths = (
            self.recording_resolution / self.center_box_to_recording_resolution_ratio
        )
        center_point_to_center_box_side_normal_lengths = center_pixel_lengths / 2.0

        x_short = (
            self.recording_center_pixel[0]
            - center_point_to_center_box_side_normal_lengths[0]
        )
        x_long = (
            self.recording_center_pixel[0]
            + center_point_to_center_box_side_normal_lengths[0]
        )
        y_short = (
            self.recording_center_pixel[1]
            + center_point_to_center_box_side_normal_lengths[1]
        )
        y_long = (
            self.recording_center_pixel[1]
            - center_point_to_center_box_side_normal_lengths[1]
        )

        return np.array(
            ((x_short, y_short), (x_short, y_long), (x_long, y_long), (x_long, y_short))
        )

    @cached_property
    def center_boolean_index(self):
        return parallel_point_in_polygon(
            self.center_square_corners[0],
            self.center_square_corners[3],
            self.center_square_corners[1],
            self.coordinates_per_frame,
            inspect=self.inspection_dir
            / CENTER_INSPECTION_DIR_NAME
            / self._inspection_image_name
            if self.inspection_dir
            else None,
        )

    @cached_property
    def periphery_boolean_index(self):
        return ~self.center_boolean_index

    @cached_property
    def motion_center(self) -> dict:
        return get_combined_features_from_merged_motion_island_data(
            self.center_boolean_index,
            self.coordinates_per_frame,
            self.meters_per_pixel,
            self.fps,
        )

    @cached_property
    def motion_periphery(self) -> dict:
        return get_combined_features_from_merged_motion_island_data(
            self.periphery_boolean_index,
            self.coordinates_per_frame,
            self.meters_per_pixel,
            self.fps,
        )

    @cached_property
    def location_sequence_center_periphery(self):
        # 1 is center, 2 is periphery, 0 is unknown
        location_sequence_center_periphery = np.zeros_like(
            self.center_boolean_index, dtype=np.uint8
        )
        location_sequence_center_periphery[self.center_boolean_index] = 1
        location_sequence_center_periphery[self.periphery_boolean_index] = 2
        return np.array(
            reduce_repeating_sequences(
                location_sequence_center_periphery,
                frame_tolerance=self._frame_tolerance,
            )
        )

    @cached_property
    def center_entries(self):
        return np.sum(self.location_sequence_center_periphery == 1)

    @cached_property
    def periphery_entries(self):
        return np.sum(self.location_sequence_center_periphery == 2)

    @cached_property
    def seconds_on_center(self):
        return np.sum(self.center_boolean_index) / self.fps

    @cached_property
    def seconds_on_periphery(self):
        return np.sum(self.periphery_boolean_index) / self.fps

    @cached_property
    def center_freezing_time(self):
        return (
            np.sum(self.motion.frozen_boolean_index & self.center_boolean_index[1:])
            / self.fps
        )

    @cached_property
    def periphery_freezing_time(self):
        return (
            np.sum(self.motion.frozen_boolean_index & self.periphery_boolean_index[1:])
            / self.fps
        )


@lru_cache
@validate_arguments
def gaussian_scoring_field(resolution: tuple[float, float], scale: int = 4):
    resolution = np.array(resolution, dtype=int) * scale

    model = ngauss_fit.model(
        x=np.indices(resolution, dtype=float),
        a=A,
        mu=resolution / 2.0,
        sigma=np.array([[resolution[0] ** 2, 0.0], [0.0, resolution[1] ** 2]]),
        axis=0,
    )

    scale_as_float = float(scale)
    return lambda x, y: model[round(x * scale_as_float)][round(y * scale_as_float)]
