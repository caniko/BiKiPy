import os
from abc import ABC
from functools import cached_property, lru_cache
from logging import getLogger
from typing import Any, ClassVar, Hashable, Optional

import numpy as np
import pandas as pd
from pydantic import Field, validate_arguments, validator, BaseModel
from pydantic_numpy import NDArray
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
    motion_multi_indexer,
)
from bikipy.perimeter.utils import perimeter_multi_indexer
from bikipy.utils.math.point_in_polygon import parallel_point_in_polygon

logger = getLogger(__name__)


A = 255
QUADRANT_INSPECTION_DIR_NAME = "PiP_quadrant_location_booleans"
CENTER_INSPECTION_DIR_NAME = "PiP_center_location_booleans"

_TWO_BY_TWO_IN_ENGLISH = {
    "upper_left": (0, 0),
    "upper_right": (0, 1),
    "lower_left": (0, 1),
    "lower_right": (1, 1),
}


class Quadrant(BikipyBase):
    corners: NDArray = Field(...)
    coordinates_per_frame: NDArray = Field(...)
    meters_per_pixel: float = Field(...)
    fps: float = Field(...)
    quadrant_index: int = Field(...)

    @validator("corners")
    def make_contiguous_array(cls, value):
        return np.ascontiguousarray(value, dtype=np.float32)

    @cached_property
    def confinement_boolean_index(self):
        return parallel_point_in_polygon(self.coordinates_per_frame, self.corners)

    @cached_property
    def seconds_present(self) -> float:
        return np.sum(self.confinement_boolean_index) / self.fps

    @cached_property
    def motion(self) -> dict:
        return get_combined_features_from_merged_motion_island_data(
            self.confinement_boolean_index,
            self.coordinates_per_frame,
            self.meters_per_pixel,
            self.fps,
        )


class RectangleEnclosedExperiment(BaseExperiment, ResolutionDerivedUnitPerPixelMixin):
    rectangle_2d_bin: tuple[int, int] = (2, 2)

    _pandas_multi_index_level: ClassVar[int] = 3

    def trial_keyword_arguments(self, trial_id: Hashable) -> dict:
        result = super().trial_keyword_arguments(trial_id)
        result["rectangle_2d_bin"] = self.rectangle_2d_bin
        return result

    @cached_property
    def motion_summary_columns(self) -> list:
        if self.recording_resolution is None:
            logger.info("Recording resolution undefined skipping center/periphery and quadrant computations")
            return super().motion_summary_columns

        quadrant_summary_columns = []
        for h in range(1, self.rectangle_2d_bin[0]):
            for v in range(1, self.rectangle_2d_bin[1]):
                quadrant_grid_coordinates = (h, v)
                quadrant_summary_columns.extend(
                    [
                        *motion_multi_indexer(quadrant_grid_coordinates, level=2),
                        *perimeter_multi_indexer(quadrant_grid_coordinates, level=2),
                    ]
                )
        return (
            super().motion_summary_columns
            + list(pd.MultiIndex.from_product([["Quadrant"], quadrant_summary_columns]))
            + list(
                pd.MultiIndex.from_product(
                    [
                        [""],
                        [
                            *motion_multi_indexer("Center", self._pandas_multi_index_level),
                            *perimeter_multi_indexer("Center", self._pandas_multi_index_level),
                            *motion_multi_indexer("Periphery", self._pandas_multi_index_level),
                            *perimeter_multi_indexer("Periphery", self._pandas_multi_index_level),
                        ],
                    ]
                )
            )
        )


class RectangleEnclosedTrial(BaseTrial, ResolutionDerivedUnitPerPixelTrialMixin, ABC):
    rectangle_2d_bin: tuple[int, int] = (2, 2)
    center_box_to_recording_resolution_ratio: Optional[float] = None
    rectangle_center_point: Optional[NDArray] = None

    @cached_property
    def _quadrant_coordinate_to_index(self):
        return {grid_coord: i for i, grid_coord in enumerate(self.quadrants, start=1)}

    @cached_property
    def _quadrant_inspection_dir(self):
        return self.inspection_dir / QUADRANT_INSPECTION_DIR_NAME

    @validator("inspection_dir")
    def make_categorical_inspection_sub_dirs(cls, value):
        if value and not (quadrant_dir := value / QUADRANT_INSPECTION_DIR_NAME).exists():
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
        scores = np.array([func(*coordinate) for coordinate in self.coordinates_per_frame])
        return np.sum(scores) / (A * self.number_of_frames)

    @cached_property
    def center_translation(self):
        return self.recording_center_pixel - self.rectangle_center_point if self.rectangle_center_point else None

    @cached_property
    def quadrants(self) -> dict[tuple[int, int], Quadrant]:
        """
        Left to right, top to down
        :return:
        """
        horizontal_uniform_distance = self.horizontal_resolution / self.rectangle_2d_bin[0]
        vertical_uniform_distance = self.vertical_resolution / self.rectangle_2d_bin[1]
        result = {}
        for quadrant_coordinate, corners in _compute_quadrant_grid_coordinates(
            self.rectangle_2d_bin, self.recording_resolution, translation=self.center_translation
        ).items():
            result[quadrant_coordinate] = Quadrant(
                corners=corners,
                coordinates_per_frame=self.coordinates_per_frame,
                meters_per_pixel=self.meters_per_pixel,
                fps=self.fps,
                quadrant_index=self._quadrant_coordinate_to_index[quadrant_coordinate],
            )
        location_sequence_quadrant = _compute_quadrant_location_sequence(result, self.number_of_frames, self.fps)
        for quadrant in result.values():
            quadrant.entries = np.sum(location_sequence_quadrant == quadrant.quadrant_index)
        return result

    @property
    def location_sequence_quadrant(self) -> NDArray:
        return _compute_quadrant_location_sequence(self.quadrants, self.number_of_frames, self.fps)

    # Center vs Periphery ==============================================================
    @cached_property
    def center_square_corners(self):
        if self.center_box_to_recording_resolution_ratio is None:
            msg = "center_box_to_recording_resolution_ratio must be defined for center and periphery analysis"
            raise AttributeError(msg)
        center_pixel_lengths = self.recording_resolution / self.center_box_to_recording_resolution_ratio
        center_point_to_center_box_side_normal_lengths = center_pixel_lengths / 2.0

        x_short = self.recording_center_pixel[0] - center_point_to_center_box_side_normal_lengths[0]
        x_long = self.recording_center_pixel[0] + center_point_to_center_box_side_normal_lengths[0]
        y_short = self.recording_center_pixel[1] + center_point_to_center_box_side_normal_lengths[1]
        y_long = self.recording_center_pixel[1] - center_point_to_center_box_side_normal_lengths[1]

        return np.array(((x_short, y_short), (x_short, y_long), (x_long, y_long), (x_long, y_short)))

    @cached_property
    def center_boolean_index(self):
        return parallel_point_in_polygon(self.coordinates_per_frame, self.center_square_corners)

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
        location_sequence_center_periphery = np.zeros_like(self.center_boolean_index, dtype=np.uint8)
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

    @property
    def motion_features(self) -> list:
        if self.recording_resolution is None:
            return super().motion_features

        return (
            super().motion_features
            + [quadrant.motion.values() for quadrant in self.quadrants.values()]
            + [
                *self.motion_center.values(),
                self.center_entries,
                self.seconds_on_center,
                *self.motion_periphery.values(),
                self.periphery_entries,
                self.seconds_on_periphery,
                self.gaussian_center_to_periphery_score,
            ]
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


@lru_cache
def _compute_quadrant_location_sequence(quadrants, number_of_frames: int, fps: float):
    result = np.zeros(number_of_frames, dtype=np.uint8)
    for i, grid_coord in enumerate(quadrants, start=1):
        result[grid_coord] = i
    return np.array(reduce_repeating_sequences(result, round(fps * 0.35)))


def _compute_quadrant_grid_coordinates(
    rectangle_2d_bin: tuple[int, int], recording_resolution: NDArray[int], translation: Optional[NDArray] = None
):
    horizontal_resolution, vertical_resolution = recording_resolution

    horizontal_uniform_distance = horizontal_resolution / rectangle_2d_bin[0]
    vertical_uniform_distance = vertical_resolution / rectangle_2d_bin[1]
    result = {}
    for h in range(1, rectangle_2d_bin[0]):
        horizontal_coordinate_min = horizontal_uniform_distance * (h - 1)
        horizontal_coordinate_max = horizontal_uniform_distance * h
        for v in range(1, rectangle_2d_bin[1]):
            vertical_coordinate_min = vertical_uniform_distance * (v - 1)
            vertical_coordinate_max = vertical_uniform_distance * v

            quadrant = np.array(
                (
                    (horizontal_coordinate_min, vertical_coordinate_min),
                    (horizontal_coordinate_max, vertical_coordinate_min),
                    (horizontal_coordinate_max, vertical_coordinate_max),
                    (horizontal_coordinate_min, vertical_coordinate_min),
                )
            )
            if translation is not None:
                quadrant += translation
            result[(h, v)] = quadrant

    return result
