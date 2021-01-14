from logging import getLogger
from typing import AnyStr, Sequence, SupportsFloat, SupportsInt, Dict

import numpy as np

from bikipy.behaviour.base import BaseExperiment
from bikipy.behaviour.nort.observation import nort_observation
from bikipy.behaviour.utils import reduce_repeating_sequences
from bikipy.border.base import PolygonalBorder, GenericPolygonalBorder
from bikipy.math.point_in_polygon import points_in_parallelogram

logger = getLogger(__name__)


class NortOpenField(BaseExperiment):
    def __init__(
        self,
        recording_resolution: Sequence[SupportsInt],
        experiment_box_real_length: SupportsFloat,
        *args,
        **kwargs,
    ):
        self.experiment_box_real_length = float(experiment_box_real_length)

        super().__init__(
            *args,
            length_unit_per_pixel=(
                self.experiment_box_real_length / np.mean(recording_resolution)
            ),
            recording_resolution=recording_resolution,
            **kwargs,
        )

        self.total_displacement = np.sum(self.displacement)

    def get_info(self):
        return [self.total_displacement, self.mean_speed, self.mean_acceleration]


class NortHabituation(NortOpenField):
    def __init__(self, center_size_real_length: SupportsFloat, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.center_size_real_length = float(center_size_real_length)
        assert self.experiment_box_real_length > self.center_size_real_length

        self.center_box_ratio = (
            (self.experiment_box_real_length - self.center_size_real_length) / 2
        ) / self.experiment_box_real_length
        self.one_minus_center_box_ratio = 1 - self.center_box_ratio

        if self.horizontal_resolution == self.vertical_resolution:
            self.center_square = (
                (  # x_short, y_long
                    self.horizontal_resolution * self.center_box_ratio,
                    self.vertical_resolution * self.one_minus_center_box_ratio,
                ),
                (  # x_short, y_short
                    self.horizontal_resolution * self.center_box_ratio,
                    self.vertical_resolution * self.center_box_ratio,
                ),
                (  # x_long, y_short
                    self.horizontal_resolution * self.one_minus_center_box_ratio,
                    self.vertical_resolution * self.center_box_ratio,
                ),
                (  # x_long, y_long
                    self.horizontal_resolution * self.one_minus_center_box_ratio,
                    self.vertical_resolution * self.one_minus_center_box_ratio,
                ),
            )
        elif self.horizontal_resolution < self.vertical_resolution:
            self.center_square = self.non_square_rectification(
                y_bias=(self.vertical_resolution - self.horizontal_resolution) / 2
            )
        else:
            self.center_square = self.non_square_rectification(
                x_bias=(self.horizontal_resolution - self.vertical_resolution) / 2
            )

        self.center_boolean_indexes = points_in_parallelogram(
            self.center_square[0],
            self.center_square[-1],
            self.center_square[1],
            self.movement_feature_coordinates,
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

        self.total_displacement = self.periphery_displacement + self.center_displacement
        self.mean_speed = (self.center_mean_speed + self.periphery_mean_speed) / 2
        self.mean_acceleration = (
            self.center_mean_acceleration + self.periphery_mean_acceleration
        ) / 2

        self.entry_sequence = np.ones_like(self.center_boolean_indexes, dtype=str)
        self.entry_sequence[self.center_boolean_indexes] = "C"
        self.entry_sequence[self.periphery_boolean_indexes] = "P"
        self.entry_sequence = np.array(reduce_repeating_sequences(self.entry_sequence))

        self.periphery_entries = np.sum(self.entry_sequence == "P")
        self.center_entries = np.sum(self.entry_sequence == "C")

    def non_square_rectification(
        self, x_bias: SupportsFloat = 0.0, y_bias: SupportsFloat = 0.0
    ):
        if (x_bias := float(x_bias)) and (y_bias := float(y_bias)):
            raise ValueError

        if x_bias:
            y_short = self.vertical_resolution * self.center_box_ratio
            y_long = self.vertical_resolution * self.one_minus_center_box_ratio

            x_short = y_short + x_bias
            x_long = y_long + x_bias

        else:
            x_short = self.horizontal_resolution * self.center_box_ratio
            x_long = self.horizontal_resolution * self.one_minus_center_box_ratio

            y_short = x_short + y_bias
            y_long = x_long + y_bias

        return (
            (x_short, y_long),
            (x_short, y_short),
            (x_long, y_short),
            (x_long, y_long),
        )

    def get_info(self):
        return super().get_info() + [
            self.periphery_displacement,
            self.periphery_mean_speed,
            self.periphery_mean_acceleration,
            self.center_displacement,
            self.center_mean_speed,
            self.center_mean_acceleration,
            self.periphery_entries,
            self.center_entries,
            self.time_in_periphery,
            self.time_in_center,
        ]


class NortWithObjects(NortHabituation):
    def __init__(
        self,
        nort_a: PolygonalBorder,
        nort_b: PolygonalBorder,
        nose_label: AnyStr,
        eye_center_label: AnyStr,
        torso_label: AnyStr,
        max_radians_gaze_and_object: SupportsFloat = 1 / 4 * np.pi,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.nort_a, self.nort_b = nort_a, nort_b
        self.torso_label, self.eye_center_label, self.nose_label = (
            str(torso_label),
            str(eye_center_label),
            str(nose_label),
        )
        self.max_radians_gaze_and_object = float(max_radians_gaze_and_object)

        self.observe_a_per_frame = self.nort_observation(self.nort_a)
        self.observe_b_per_frame = self.nort_observation(self.nort_b)
        self.not_observing = np.logical_not(
            np.logical_or(self.observe_a_per_frame, self.observe_b_per_frame)
        )

        assert self.observe_a_per_frame.size == self.observe_b_per_frame.size

        self.observation_sequence = np.ones_like(self.observe_a_per_frame, dtype=str)

        self.observation_sequence[self.observe_a_per_frame] = "A"
        self.observation_sequence[self.observe_b_per_frame] = "B"
        self.observation_sequence[self.not_observing] = "X"

        self.reduced_observation_sequence = np.array(
            reduce_repeating_sequences(self.observation_sequence)
        )

        self.novelty_observation_a = np.sum(self.reduced_observation_sequence == "A")
        self.novelty_observation_b = np.sum(self.reduced_observation_sequence == "B")

        self.time_spent_a = np.sum(self.observation_sequence == "A") / self.fps
        self.time_spent_b = np.sum(self.observation_sequence == "B") / self.fps

    def get_info(self):
        return super().get_info() + [
            self.novelty_observation_a,
            self.novelty_observation_b,
            self.time_spent_a,
            self.time_spent_b,
            self.time_spent_a + self.time_spent_b,
        ]

    def nort_observation(self, nort_object):
        return nort_observation(
            nort_object,
            self.coordinate_sequence[self.eye_center_label],
            self.coordinate_sequence[self.nose_label],
            self.coordinate_sequence[self.torso_label],
            self.fps,
            self.max_radians_gaze_and_object,
        )


class NortObjectField:
    def __init__(
        self,
        constant_object: PolygonalBorder,
        variable_object: PolygonalBorder,
        novel_object: PolygonalBorder,
    ):
        self.constant_object = constant_object
        self.variable_object = variable_object
        self.novel_object = novel_object

    @classmethod
    def from_images(cls, habituation_img, novelty_img):
        GenericPolygonalBorder.corners = 3

        habituation_border_a = GenericPolygonalBorder.from_image(habituation_img)
        habituation_border_b = GenericPolygonalBorder.from_image(habituation_img)

        novel_object = GenericPolygonalBorder.from_image(novelty_img)

        if GenericPolygonalBorder.distance_between_two_borders(
            habituation_border_a, novel_object
        ) < GenericPolygonalBorder.distance_between_two_borders(
            habituation_border_b, novel_object
        ):
            constant_border = habituation_border_b
            variable_border = habituation_border_a
        else:
            constant_border = habituation_border_a
            variable_border = habituation_border_b

        return cls(constant_border, variable_border, novel_object)

    def habituation(self, *args, **kwargs):
        experiment = NortWithObjects(
            nort_a=self.nort_identical_obj,
            nort_b=self.nort_identical_obj_variable,
            *args,
            **kwargs
        )



    def novelty(self, *args, **kwargs):
        experiment = NortWithObjects(
            nort_a=self.nort_identical_obj,
            nort_b=self.nort_novel,
            *args,
            **kwargs
        )


