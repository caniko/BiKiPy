"""
Novel Object Recognition test (NORT) class representing a single test.
These tests can be grouped together to form entire experiments.
"""

from logging import getLogger
from typing import Any, AnyStr, Sequence, SupportsFloat, SupportsInt, Union

import matplotlib.pyplot as plt
import numpy as np

from bikipy.behaviour.base import BaseTrial
from bikipy.behaviour.nort.observation import nort_observation
from bikipy.behaviour.utils import reduce_repeating_sequences
from bikipy.border.base import GenericPolygonalBorder, PolygonalBorder
from bikipy.math.point_in_polygon import points_in_parallelogram

logger = getLogger(__name__)


class NortHabituation(BaseTrial):
    """
    NORT experiment without any objects. The purpose of this test is to generate
    reference data for future NORT experiments.
    """

    def __init__(
        self,
        recording_resolution: Sequence[SupportsInt],
        experiment_box_real_length: SupportsFloat,
        center_size_real_length: SupportsFloat,
        *base_experiment_args,
        **base_experiment_kwargs,
    ):
        """
        Parameters
        ----------
        recording_resolution: array_like
            Resolution of the video used to record the experiment
        experiment_box_real_length: float
            Length of the square box in which the experiment is conducted
        center_size_real_length: float
            Length of the square box signifying periphery and inner area of the
            square box
        base_experiment_args
            Arguments passed to BaseTrial
        base_experiment_kwargs
            Keyword arguments passed to BaseTrial
        """
        self.experiment_box_real_length = float(experiment_box_real_length)

        super().__init__(
            *base_experiment_args,
            length_unit_per_pixel=(
                self.experiment_box_real_length / np.mean(recording_resolution)
            ),
            recording_resolution=recording_resolution,
            **base_experiment_kwargs,
        )

        self.total_displacement = np.sum(self.displacement)

        self.center_size_real_length = float(center_size_real_length)
        assert self.experiment_box_real_length > self.center_size_real_length

        self.center_box_ratio = (
            (self.experiment_box_real_length - self.center_size_real_length) / 2
        ) / self.experiment_box_real_length
        self.one_minus_center_box_ratio = 1 - self.center_box_ratio

        x = self.horizontal_resolution * self.center_box_ratio
        x_rest_half = (self.horizontal_resolution - x) / 2

        y = self.vertical_resolution * self.center_box_ratio
        y_rest_half = (self.vertical_resolution - y) / 2
        if self.horizontal_resolution == self.vertical_resolution:
            self.center_square = (
                # x_short, y_long
                (x_rest_half, y_rest_half),
                # x_short, y_short
                (x_rest_half, self.vertical_resolution - y_rest_half),
                # x_long, y_short
                (
                    self.horizontal_resolution - x_rest_half,
                    self.vertical_resolution - y_rest_half,
                ),
                # x_long, y_long
                (self.horizontal_resolution - x_rest_half, y_rest_half),
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
            self.coordinates_per_frame,
        )
        self.periphery_boolean_indexes = np.logical_and(
            np.logical_not(self.center_boolean_indexes),
            np.logical_and(*np.isfinite(self.coordinates_per_frame).T),
        )

        self.seconds_in_center = np.sum(self.center_boolean_indexes) / self.fps
        self.seconds_in_periphery = np.sum(self.periphery_boolean_indexes) / self.fps

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
        return [
            self.total_displacement,
            self.mean_speed,
            self.mean_acceleration,
            self.periphery_displacement,
            self.periphery_mean_speed,
            self.periphery_mean_acceleration,
            self.center_displacement,
            self.center_mean_speed,
            self.center_mean_acceleration,
            self.periphery_entries,
            self.center_entries,
            self.seconds_in_periphery,
            self.seconds_in_center,
        ]

    def plot(self, ax: Any = None):
        if not ax:
            fig, ax = plt.subplots()
        if self.guiding_image is not None:
            ax.imshow(self.guiding_image)
        else:
            logger.warning("guiding_image is not defined will plot without")

        for i in range((max := len(self.center_square))):
            next_i = i + 1
            ax.plot(
                self.center_square[i],
                self.center_square[next_i if next_i != max else 0],
            )

        return ax


class NortOpenField(NortHabituation):
    pass


class NortObjectTraining(NortHabituation):
    def __init__(
        self,
        nort_a: PolygonalBorder,
        nort_b: PolygonalBorder,
        nose_label: AnyStr,
        eye_center_label: AnyStr,
        torso_label: AnyStr,
        max_radians_gaze_and_object: SupportsFloat = 1 / 4 * np.pi,
        *nort_habituation_args,
        **nort_habituation_kwargs,
    ):
        super().__init__(*nort_habituation_args, **nort_habituation_kwargs)

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

        self.observation_sequence = np.zeros_like(
            self.observe_a_per_frame, dtype=np.str
        )

        self.observation_sequence[self.observe_a_per_frame] = "A"
        self.observation_sequence[self.observe_b_per_frame] = "B"
        self.observation_sequence[self.not_observing] = "X"

        self.reduced_observation_sequence = np.array(
            reduce_repeating_sequences(self.observation_sequence)
        )

        self.novelty_observation_a = np.sum(self.reduced_observation_sequence == "A")
        self.novelty_observation_b = np.sum(self.reduced_observation_sequence == "B")

        self.seconds_spent_a = np.sum(self.observation_sequence == "A") / self.fps
        self.seconds_spent_b = np.sum(self.observation_sequence == "B") / self.fps
        self.seconds_observing = self.seconds_spent_a + self.seconds_spent_b

        self.object_bias_score = 100 * self.seconds_spent_a / self.seconds_observing

        assert (
            self.seconds_observing < self.experiment_seconds
        ), f"not {self.seconds_observing} < {self.experiment_seconds}"

    def get_info(self):
        return super().get_info() + [
            self.novelty_observation_a,
            self.novelty_observation_b,
            self.seconds_spent_a,
            self.seconds_spent_b,
            self.seconds_observing,
            self.object_bias_score,
        ]

    def nort_observation(self, nort_object):
        eye, nose, torso = self.coordinate_sequence[
            self.eye_center_label, self.nose_label, self.torso_label
        ]
        return nort_observation(
            nort_object,
            eye,
            nose,
            torso,
            self.fps,
            self.max_radians_gaze_and_object,
            inspection_image=self.guiding_image,
        )

    def plot(self, ax: Any = None):
        ax = super().plot(ax)

        self.nort_a.plot(ax=ax, include_borders=True)
        self.nort_b.plot(ax=ax, include_borders=True)

        return ax


class NortNovelObject(NortObjectTraining):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.absolute_discrimination = np.sum(self.observe_b_per_frame) - np.sum(
            self.observe_a_per_frame
        )

        self.discrimination_index = (
            self.absolute_discrimination / self.experiment_seconds
        )

        self.novelty_preference = 100 * self.seconds_spent_b / self.experiment_seconds

    def get_info(self):
        return super().get_info() + [
            self.absolute_discrimination,
            self.discrimination_index,
            self.novelty_preference,
        ]


class NortObjectField:
    def __init__(
        self,
        constant_object: GenericPolygonalBorder,
        variable_object: GenericPolygonalBorder,
        novel_object: GenericPolygonalBorder,
        label: Union[AnyStr, None] = None,
    ):
        assert (
            constant_object.border_distance
            and variable_object.border_distance
            and novel_object.border_distance
        )

        self.constant_object = constant_object
        self.variable_object = variable_object
        self.novel_object = novel_object
        self.label = str(label)

    @classmethod
    def from_images(
        cls,
        habituation_img: Any,
        novelty_img: Any,
        border_distance: SupportsFloat,
        **kwargs,
    ):
        GenericPolygonalBorder.corners = 4

        habituation_border_a = GenericPolygonalBorder.from_image(
            habituation_img, border_distance=border_distance
        )
        habituation_border_b = GenericPolygonalBorder.from_image(
            habituation_img, border_distance=border_distance
        )

        novel_object = GenericPolygonalBorder.from_image(
            novelty_img, border_distance=border_distance, semantic_label="novel"
        )

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

        constant_border.semantic_label = "constant"
        variable_border.semantic_label = "variable"

        return cls(constant_border, variable_border, novel_object, **kwargs)

    def habituation(self, *args, **kwargs):
        return NortObjectTraining(
            nort_a=self.constant_object,
            nort_b=self.variable_object,
            *args,
            **kwargs,
        )

    def training(self, *args, **kwargs):
        return self.habituation(*args, **kwargs)

    def novelty(self, *args, **kwargs):
        return NortObjectTraining(
            nort_a=self.constant_object, nort_b=self.novel_object, *args, **kwargs
        )


class NortAnimalData:
    # TODO: WIP
    def __init__(self, habituation, novelty, animal_id):
        self.animal_id = int(animal_id)

        self.habituation = habituation
        self.novelty = novelty
