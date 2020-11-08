from typing import Union, AnyStr, Any, SupportsFloat, SupportsInt, Sequence
from warnings import warn
import itertools as it

import pandas as pd
import numpy as np

from bikipy.math.point_in_polygon import points_in_parallelogram
from bikipy.border.base import PolygonalBorder

from bikipy.behaviour.nort.observation import nort_observation
from bikipy.behaviour.base import BaseExperiment
from bikipy.behavirou.utils import reduce_str_sequence


class NortBase(BaseExperiment):
    def __init__(
        self,
        coordinate_sequence: Any,
        recording_resolution: Sequence[SupportsInt],
        square_box_size_cm: SupportsFloat,
        center_size_cm: SupportsFloat,
        fps: SupportsFloat,
        cm_per_pixel: SupportsFloat,
        movement_feature_point_label: Union[AnyStr, None] = None,
        label: Any = None,
    ):
        super().__init__(
            coordinate_sequence, fps, cm_per_pixel, movement_feature_point_label, label
        )

        self.x_res = int(recording_resolution[0])
        self.y_res = int(recording_resolution[1])

        self.square_box_size_cm = float(square_box_size_cm)
        self.center_size_cm = float(center_size_cm)
        assert self.square_box_size_cm > self.center_size_cm

        self.center_box_ratio = self.center_size_cm / self.square_box_size_cm
        self.one_minus_center_box_ratio = 1 - self.center_box_ratio

        if self.x_res == self.y_res:
            self.center_square = (
                (  # x_short, y_long
                    self.x_res * self.center_box_ratio,
                    self.y_res * self.one_minus_center_box_ratio,
                ),
                (  # x_short, y_short
                    self.x_res * self.center_box_ratio,
                    self.y_res * self.center_box_ratio,
                ),
                (  # x_long, y_short
                    self.x_res * self.one_minus_center_box_ratio,
                    self.y_res * self.center_box_ratio,
                ),
                (  # x_long, y_long
                    self.x_res * self.one_minus_center_box_ratio,
                    self.y_res * self.one_minus_center_box_ratio,
                ),
            )
        elif self.x_res < self.y_res:
            self.center_square = self.non_square_rectification(
                y_bias=(self.y_res - self.x_res) / 2
            )
        else:
            self.center_square = self.non_square_rectification(
                x_bias=(self.x_res - self.y_res) / 2
            )

        self.center_boolean_indexes = points_in_parallelogram(
            self.center_square[0],
            self.center_square[-1],
            self.center_square[1],
            self.coordinate_sequence,
        )
        self.periphery_boolean_indexes = np.logical_not(self.center_boolean_indexes)

        self.time_in_center = np.sum(self.center_boolean_indexes) / self.fps
        self.time_in_periphery = np.sum(self.periphery_boolean_indexes) / self.fps

        (
            self.center_displacement,
            self.center_mean_speed,
            self.center_mean_acceleration,
        ) = self.compute_movement_features_over_boolean_index(
            self.center_boolean_indexes
        )
        (
            self.periphery_displacement,
            self.periphery_mean_speed,
            self.periphery_mean_acceleration,
        ) = self.compute_movement_features_over_boolean_index(
            self.periphery_boolean_indexes
        )

        self.entry_sequence = np.zeros_like(self.center_boolean_indexes, dtype=str)
        self.entry_sequence[self.center_boolean_indexes] = "C"
        self.entry_sequence[self.periphery_displacement] = "P"
        self.entry_sequence = reduce_str_sequence(self.entry_sequence)

    def non_square_rectification(
        self, x_bias: SupportsFloat = 0.0, y_bias: SupportsFloat = 0.0
    ):
        if (x_bias := float(x_bias)) and (y_bias := float(y_bias)):
            raise ValueError

        if x_bias:
            y_short = self.y_res * self.center_box_ratio
            y_long = self.y_res * self.one_minus_center_box_ratio

            x_short = y_short + x_bias
            x_long = y_long + x_bias

        else:
            x_short = self.x_res * self.center_box_ratio
            x_long = self.x_res * self.one_minus_center_box_ratio

            y_short = x_short + y_bias
            y_long = x_long + y_bias

        return (
            (x_short, y_long),
            (x_short, y_short),
            (x_long, y_short),
            (x_long, y_long),
        )


class NortHabituation(NortBase):
    pass


class NortWithObjects(NortBase):
    def __init__(
        self,
        nort_a: PolygonalBorder,
        nort_b: PolygonalBorder,
        coordinate_sequence: Any,
        fps: SupportsFloat,
        cm_per_pixel: SupportsFloat,
        movement_feature_point_label: AnyStr,
        max_radians_gaze_and_object: SupportsFloat = 1 / 4 * np.pi,
    ):
        super().__init__(
            coordinate_sequence, fps, cm_per_pixel, movement_feature_point_label
        )

        self.nort_a, self.nort_b = nort_a, nort_b

        self.max_radians_gaze_and_object = float(max_radians_gaze_and_object)

        self.observe_times_a, self.observe_a_start_end = self._dlc_nort_observation(
            self.nort_a
        )
        self.observe_times_b, self.observe_b_start_end = self._dlc_nort_observation(
            self.nort_b
        )

    def _dlc_nort_observation(self, nort_object):
        return nort_observation(
            nort_object,
            self.coordinate_sequence["mid-left_ear-right_ear"],
            self.coordinate_sequence["nose"],
            self.coordinate_sequence["mid-mid-left_ear-right_ear-tail"],
            self.fps,
            self.max_radians_gaze_and_object,
        )
