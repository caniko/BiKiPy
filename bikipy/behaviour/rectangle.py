from abc import ABC
from functools import cached_property

import numpy as np

from bikipy.behaviour.base import BaseExperiment, BaseTrial
from bikipy.behaviour.mixins.unit_per_pixel.resolution_derived import \
    ResolutionDerivedUnitPerPixelMixin
from bikipy.behaviour.utils import reduce_repeating_sequences
from bikipy.feature.motion import (
    get_combined_features_from_merged_motion_island_data,
    motion_2d_multi_indexer,
)
from bikipy.math.point_in_polygon import points_in_parallelogram


class RectangleEnclosedExperiment(ResolutionDerivedUnitPerPixelMixin, BaseExperiment):
    @cached_property
    def _motion_summary_columns(self) -> list:
        quadrant_labels = (
            "Upper-left Quadrant",
            "Upper-right Quadrant",
            "Lower-left Quadrant",
            "Lower-right Quadrant",
        )
        return super()._motion_summary_columns + [
            *motion_2d_multi_indexer("Upper-left Quadrant"),
            *motion_2d_multi_indexer("Upper-right Quadrant"),
            *motion_2d_multi_indexer("Lower-left Quadrant"),
            *motion_2d_multi_indexer("Lower-right Quadrant"),
            *self._feature_2d_multi_indexer("Seconds present", quadrant_labels),
            *self._feature_2d_multi_indexer("Entries", quadrant_labels),
        ]


class RectangleEnclosedTrial(ResolutionDerivedUnitPerPixelMixin, BaseTrial, ABC):
    @cached_property
    def location_sequence_quadrant(self) -> np.ndarray:
        result = self._uint_zeros_based_on_frame_length.copy()

        result[self.quadrant_upper_left_boolean_index] = 1
        result[self.quadrant_upper_right_boolean_index] = 2
        result[self.quadrant_lower_left_boolean_index] = 3
        result[self.quadrant_lower_right_boolean_index] = 4

        return reduce_repeating_sequences(result, round(self.fps * 0.35))

    @cached_property
    def motion_quadrant_upper_left(self) -> dict:
        return get_combined_features_from_merged_motion_island_data(
            self.quadrant_upper_left_boolean_index,
            self.coordinates_per_frame,
            self.units_per_pixel,
            self.fps,
        )

    @cached_property
    def motion_quadrant_upper_right(self) -> dict:
        return get_combined_features_from_merged_motion_island_data(
            self.quadrant_upper_right_boolean_index,
            self.coordinates_per_frame,
            self.units_per_pixel,
            self.fps,
        )

    @cached_property
    def motion_quadrant_lower_left(self) -> dict:
        return get_combined_features_from_merged_motion_island_data(
            self.quadrant_lower_left_boolean_index,
            self.coordinates_per_frame,
            self.units_per_pixel,
            self.fps,
        )

    @cached_property
    def motion_quadrant_lower_right(self) -> dict:
        return get_combined_features_from_merged_motion_island_data(
            self.quadrant_lower_right_boolean_index,
            self.coordinates_per_frame,
            self.units_per_pixel,
            self.fps,
        )

    # Quadrant upper left 1

    @cached_property
    def quadrant_upper_left_boolean_index(self) -> np.ndarray:
        return points_in_parallelogram(
            np.array((0.0, 0.0)),
            np.array((self.recording_center_pixel[0], 0.0)),
            np.array((0.0, self.recording_center_pixel[1])),
            self.coordinates_per_frame,
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
        return points_in_parallelogram(
            np.array((self.recording_center_pixel[0], 0.0)),
            self.recording_center_pixel,
            np.array((self.horizontal_resolution, 0.0)),
            self.coordinates_per_frame,
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
        return points_in_parallelogram(
            np.array((0.0, self.vertical_resolution)),
            np.array((self.recording_center_pixel[0], self.vertical_resolution)),
            np.array((0.0, self.recording_center_pixel[1])),
            self.coordinates_per_frame,
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
        return points_in_parallelogram(
            np.array((self.recording_center_pixel[0], self.vertical_resolution)),
            self.recording_resolution,
            self.recording_center_pixel,
            self.coordinates_per_frame,
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
