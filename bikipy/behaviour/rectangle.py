import os
from functools import cached_property, lru_cache
from logging import getLogger
from typing import ClassVar, Hashable, Optional

import numpy as np
import pandas as pd
from pydantic import validate_arguments, validator
from skg import ngauss_fit

from bikipy.behaviour.base import BaseExperiment, BaseTrial
from bikipy.behaviour.utils import reduce_repeating_sequences
from bikipy.core.base_class import BikipyBase
from bikipy.core.typing import NDArrayBool, NDArrayFp64
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
    corners: NDArrayFp64
    framewise_confined_coordinates: NDArrayFp64
    meters_per_pixel: float
    fps: float
    quadrant_index: int

    @cached_property
    def confinement_boolean_index(self) -> NDArrayBool:
        return parallel_point_in_polygon(self.framewise_confined_coordinates, self.corners)

    @cached_property
    def seconds_present(self) -> float:
        return np.sum(self.confinement_boolean_index) / self.fps

    @cached_property
    def motion(self) -> dict:
        return get_combined_features_from_merged_motion_island_data(
            self.confinement_boolean_index,
            self.framewise_confined_coordinates,
            self.meters_per_pixel,
            self.fps,
        )


class RectangleEnclosedExperiment(BaseExperiment):
    rectangle_2d_bin: tuple[int, int] = (2, 2)
    center_box_to_recording_resolution_ratio: Optional[float] = None

    def trial_keyword_arguments(self, trial_id: Hashable) -> dict:
        result = super().trial_keyword_arguments(trial_id)
        result["rectangle_2d_bin"] = self.rectangle_2d_bin
        result["center_box_to_recording_resolution_ratio"] = self.center_box_to_recording_resolution_ratio
        return result

    @property
    def quadrant_grid_coordinates(self):
        result = []
        for h in range(1, self.rectangle_2d_bin[0] + 1):
            for v in range(1, self.rectangle_2d_bin[1] + 1):
                result.append((h, v))
        return result

    @cached_property
    def motion_summary_columns(self) -> list:
        quadrant_summary_columns = []
        for h in range(1, self.rectangle_2d_bin[0]+1):
            for v in range(1, self.rectangle_2d_bin[1]+1):
                quadrant_grid_coordinates = (h, v)
                quadrant_summary_columns.extend(
                    [
                        *motion_multi_indexer(quadrant_grid_coordinates, level=self._pandas_multi_index_level),
                        *perimeter_multi_indexer(quadrant_grid_coordinates, level=self._pandas_multi_index_level),
                    ]
                )
        result = (
            super().motion_summary_columns
            + [["Gaussian", "CenterToPeriphery"]]
            + list(pd.MultiIndex.from_product([["Quadrant"], quadrant_summary_columns]))
            + list(pd.MultiIndex.from_product([["QuadrantEntries"], self.quadrant_grid_coordinates]))
        )
        if self.center_box_to_recording_resolution_ratio:
            result += list(
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
        return result


class RectangleEnclosedTrial(BaseTrial):
    rectangle_2d_bin: tuple[int, int] = (2, 2)
    center_box_to_recording_resolution_ratio: Optional[float] = None

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
        scores = np.array(
            [func(*coordinate) for coordinate in self.framewise_confined_coordinates if not np.any(np.isnan(coordinate))]
        )
        return np.sum(scores) / (A * self.number_of_frames)

    @cached_property
    def quadrant_grid_coordinate_to_corners(self):
        horizontal_uniform_distance = self.horizontal_resolution / self.rectangle_2d_bin[0]
        vertical_uniform_distance = self.vertical_resolution / self.rectangle_2d_bin[1]
        result = {}
        for h in range(1, self.rectangle_2d_bin[0]):
            horizontal_coordinate_min = horizontal_uniform_distance * (h - 1)
            horizontal_coordinate_max = horizontal_uniform_distance * h
            for v in range(1, self.rectangle_2d_bin[1]):
                vertical_coordinate_min = vertical_uniform_distance * (v - 1)
                vertical_coordinate_max = vertical_uniform_distance * v

                quadrant = np.array(
                    (
                        (horizontal_coordinate_min, vertical_coordinate_min),
                        (horizontal_coordinate_max, vertical_coordinate_min),
                        (horizontal_coordinate_max, vertical_coordinate_max),
                        (horizontal_coordinate_min, vertical_coordinate_max),
                    )
                )
                if self.center_translation is not None:
                    quadrant += self.center_translation
                result[(h, v)] = quadrant
        return result

    @cached_property
    def quadrant_index_to_quadrant_grid_coordinate(self):
        return {
            i: quadrant_grid_coordinate
            for i, quadrant_grid_coordinate in enumerate(self.quadrant_grid_coordinate_to_corners, start=1)
        }

    @cached_property
    def quadrant_grid_coordinate_to_quadrant_index(self):
        return {
            quadrant_grid_coordinate: i
            for i, quadrant_grid_coordinate in self.quadrant_index_to_quadrant_grid_coordinate.items()
        }

    @cached_property
    def quadrant_grid_coordinate_to_quadrant(self) -> dict[tuple[int, int], Quadrant]:
        """
        Left to right, top to down
        :return:
        """
        result = {}
        for quadrant_index, quadrant_grid_coordinate in self.quadrant_index_to_quadrant_grid_coordinate.items():
            result[quadrant_grid_coordinate] = Quadrant(
                corners=self.quadrant_grid_coordinate_to_corners[quadrant_grid_coordinate],
                framewise_confined_coordinates=self.framewise_confined_coordinates,
                meters_per_pixel=self.meters_per_pixel,
                fps=self.fps,
                quadrant_index=quadrant_index,
            )
        return result

    @cached_property
    def location_sequence_quadrant(self) -> NDArrayFp64:
        raw_location_sequence_quadrant = np.zeros(self.number_of_frames, dtype=np.uint8)
        for quadrant_index, quadrant_grid_coordinate in self.quadrant_index_to_quadrant_grid_coordinate.items():
            quadrant = self.quadrant_grid_coordinate_to_quadrant[quadrant_grid_coordinate]

            if np.any(raw_location_sequence_quadrant[quadrant.confinement_boolean_index]):
                logger.warning("Quadrant confinement has temporal-spatial collision with another, ignoring")

            raw_location_sequence_quadrant[quadrant.confinement_boolean_index] = quadrant_index

        return np.array(reduce_repeating_sequences(raw_location_sequence_quadrant, round(self.fps * 0.35)))

    @cached_property
    def quadrant_grid_coordinate_to_entries(self):
        result = {}
        for quadrant_index, quadrant_grid_coordinate in self.quadrant_index_to_quadrant_grid_coordinate.items():
            result[quadrant_grid_coordinate] = np.sum(self.location_sequence_quadrant == quadrant_index)
        return result

    # Center vs Periphery ==============================================================
    @cached_property
    def center_rectangle_corners(self) -> NDArrayFp64:
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
    def center_boolean_index(self) -> NDArrayBool:
        return parallel_point_in_polygon(self.framewise_confined_coordinates, self.center_rectangle_corners)

    @cached_property
    def periphery_boolean_index(self) -> NDArrayBool:
        return ~self.center_boolean_index

    @cached_property
    def motion_center(self) -> dict:
        return get_combined_features_from_merged_motion_island_data(
            self.center_boolean_index,
            self.framewise_confined_coordinates,
            self.meters_per_pixel,
            self.fps,
        )

    @cached_property
    def motion_periphery(self) -> dict:
        return get_combined_features_from_merged_motion_island_data(
            self.periphery_boolean_index,
            self.framewise_confined_coordinates,
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

        quadrant_motion_values = []
        for quadrant in self.quadrant_grid_coordinate_to_quadrant.values():
            quadrant_motion_values.extend(list(quadrant.motion.values()))

        result = [
            *super().motion_features,
            self.gaussian_center_to_periphery_score,
            *quadrant_motion_values,
            *self.quadrant_grid_coordinate_to_entries.values()
        ]

        if self.center_box_to_recording_resolution_ratio:
            result.extend(
                [
                    *self.motion_center.values(),
                    self.center_entries,
                    self.seconds_on_center,
                    *self.motion_periphery.values(),
                    self.periphery_entries,
                    self.seconds_on_periphery,
                ]
            )
        return result


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
    rectangle_2d_bin: tuple[int, int], recording_resolution: tuple[int, int], translation: Optional[NDArrayFp64] = None
) -> NDArrayFp64:
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
