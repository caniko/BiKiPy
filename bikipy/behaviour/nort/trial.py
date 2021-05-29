"""
Novel Object Recognition test (NORT) class representing a single test.
These tests can be grouped together to form entire experiments.
"""
from dataclasses import dataclass, field
from logging import getLogger
from pathlib import Path
from typing import Any, SupportsFloat, Iterable, Union

import matplotlib.pyplot as plt
import numpy as np
from compress_pickle import compress_pickle

from bikipy.behaviour.base import BaseTrial
from bikipy.behaviour.nort.observation import nort_observation
from bikipy.behaviour.utils import python_reduce_repeating_sequences
from bikipy.perimeter.base import PolygonalPerimeter
from bikipy.math.point_in_polygon import points_in_parallelogram

logger = getLogger(__name__)


class NortHabituationTrial(BaseTrial):
    """
    NORT experiment without any objects. The purpose of this test is to generate
    reference data for future NORT experiments.
    """

    def __init__(
        self,
        recording_resolution: Iterable[int],
        experiment_box_metric_length: SupportsFloat,
        center_size_metric_length: SupportsFloat,
        *base_experiment_args,
        **base_experiment_kwargs,
    ):
        """
        Parameters
        ----------
        recording_resolution: array_like
            Resolution of the video used to record the experiment
        experiment_box_metric_length: float
            Length of the square box in which the experiment is conducted
        center_size_metric_length: float
            Length of the square box signifying periphery and inner area of the
            square box
        base_experiment_args
            Arguments passed to BaseTrial
        base_experiment_kwargs
            Keyword arguments passed to BaseTrial
        """
        self.experiment_box_metric_length = float(experiment_box_metric_length)

        super().__init__(
            *base_experiment_args,
            length_unit_per_pixel=(
                self.experiment_box_metric_length / min(recording_resolution)
            ),
            recording_resolution=recording_resolution,
            **base_experiment_kwargs,
        )

        self.total_displacement = np.sum(self.displacement)

        self.center_size_metric_length = float(center_size_metric_length)
        assert self.experiment_box_metric_length > self.center_size_metric_length

        self.center_box_ratio = (
            (self.experiment_box_metric_length - self.center_size_metric_length) / 2.0
        ) / self.experiment_box_metric_length
        self.one_minus_center_box_ratio = 1.0 - self.center_box_ratio

        x = self.horizontal_resolution * self.center_box_ratio
        x_rest_half = (self.horizontal_resolution - x) / 2.0

        y = self.vertical_resolution * self.center_box_ratio
        y_rest_half = (self.vertical_resolution - y) / 2.0
        if self.horizontal_resolution == self.vertical_resolution:
            self.center_square = (
                # x_short, y_short
                (x_rest_half, y_rest_half),
                # x_short, y_long
                (x_rest_half, self.vertical_resolution - y_rest_half),
                # x_long, y_long
                (
                    self.horizontal_resolution - x_rest_half,
                    self.vertical_resolution - y_rest_half,
                ),
                # x_long, y_short
                (self.horizontal_resolution - x_rest_half, y_rest_half),
            )
        elif self.horizontal_resolution < self.vertical_resolution:
            self.center_square = self.non_square_rectification(
                y_bias=(self.vertical_resolution - self.horizontal_resolution) / 2.0
            )
        else:
            self.center_square = self.non_square_rectification(
                x_bias=(self.horizontal_resolution - self.vertical_resolution) / 2.0
            )

        self.center_boolean_indexes = points_in_parallelogram(
            self.center_square[0],
            self.center_square[3],
            self.center_square[1],
            self.coordinates_per_frame,
            inspect_points=self.func_inspect,
        )
        self.periphery_boolean_indexes = np.logical_and(
            ~self.center_boolean_indexes,
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
        if not self.center_displacement or self.center_displacement == 0:
            self.center_boolean_indexes = points_in_parallelogram(
                self.center_square[0],
                self.center_square[3],
                self.center_square[1],
                self.coordinates_per_frame,
                inspect_points=self.func_inspect,
            )
        (
            self.periphery_displacement,
            self.periphery_mean_speed,
            self.periphery_mean_acceleration,
        ) = self.compute_movement_features_over_boolean_index(
            self.periphery_boolean_indexes
        )

        self.total_displacement = self.periphery_displacement + self.center_displacement
        self.mean_speed = (self.center_mean_speed + self.periphery_mean_speed) / 2.0
        self.mean_acceleration = (
            self.center_mean_acceleration + self.periphery_mean_acceleration
        ) / 2.0

        # 1 is center, 2 is periphery, 0 is invalid aka unknown
        self.location_sequence = np.zeros_like(
            self.center_boolean_indexes, dtype=np.uint8
        )
        self.location_sequence[self.center_boolean_indexes] = 1
        self.location_sequence[self.periphery_boolean_indexes] = 2
        self.location_sequence = np.array(
            python_reduce_repeating_sequences(self.location_sequence)
        )

        self.center_entries = np.sum(self.location_sequence == 1)
        self.periphery_entries = np.sum(self.location_sequence == 2)

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
            (x_short, y_short),
            (x_short, y_long),
            (x_long, y_long),
            (x_long, y_short),
        )

    def info(self):
        return [
            self.label,
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
        if self.inspect_image is not None:
            ax.imshow(self.inspect_image)
        else:
            logger.warning("inspect_image is not defined will plot without")

        for i in range((max := len(self.center_square))):
            next_i = i + 1
            ax.plot(
                self.center_square[i],
                self.center_square[next_i if next_i != max else 0],
            )

        return ax


class NortOpenField(NortHabituationTrial):
    pass


class NortTrainingTrial(NortHabituationTrial):
    def __init__(
        self,
        nort_a: PolygonalPerimeter,
        nort_b: PolygonalPerimeter,
        nose_label: str,
        eye_center_label: str,
        torso_label: str,
        perimeter_border_normal_metric_magnitude: SupportsFloat,
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
        self.perimeter_border_normal_metric_magnitude = float(
            perimeter_border_normal_metric_magnitude
        )
        self.perimeter_border_normal_pixel_magnitude = (
            self.perimeter_border_normal_metric_magnitude / self.length_unit_per_pixel
        )

        (
            self.a_observance_per_frame,
            self.a_location_filtered,
            self.a_gaze_filtered,
        ) = self.nort_observation(self.nort_a)
        (
            self.b_observance_per_frame,
            self.b_location_filtered,
            self.b_gaze_filtered,
        ) = self.nort_observation(self.nort_b)

        self.not_observing = ~(
            self.a_observance_per_frame | self.b_observance_per_frame
        )

        assert self.a_observance_per_frame.size == self.b_observance_per_frame.size

        self.observation_sequence = np.zeros_like(
            self.a_observance_per_frame, dtype=np.uint8
        )

        self.observation_sequence[self.a_observance_per_frame] = 1
        self.observation_sequence[self.b_observance_per_frame] = 2
        # assert np.all((self.observation_sequence == 0) == self.not_observing)

        self.reduced_observation_sequence = np.array(
            python_reduce_repeating_sequences(self.observation_sequence)
        )

        self.novelty_observation_a = np.sum(self.reduced_observation_sequence == 1)
        self.novelty_observation_b = np.sum(self.reduced_observation_sequence == 2)

        self.seconds_spent_a = np.sum(self.observation_sequence == 1) / self.fps
        self.seconds_spent_b = np.sum(self.observation_sequence == 2) / self.fps
        self.seconds_observing = self.seconds_spent_a + self.seconds_spent_b

        self.object_bias_score = (
            100.0 * self.seconds_spent_a / self.seconds_observing
            if self.seconds_observing
            else 0
        )

        assert (
            self.seconds_observing < self.experiment_seconds
        ), f"{self.seconds_observing} > {self.experiment_seconds}"

    def info(self):
        return super().info() + [
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
            self.perimeter_border_normal_pixel_magnitude,
            inspect=self.func_inspect,
            inspection_image=self.inspect_image,
        )

    def plot(self, ax: Any = None):
        ax = super().plot(ax)

        self.nort_a.plot(ax=ax)
        self.nort_b.plot(ax=ax)

        return ax


class NortNoveltyTrial(NortTrainingTrial):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.absolute_discrimination = np.sum(self.b_observance_per_frame) - np.sum(
            self.a_observance_per_frame
        )

        self.discrimination_index = (
            self.absolute_discrimination / self.experiment_seconds
        )

        self.novelty_preference = 100 * self.seconds_spent_b / self.experiment_seconds

    def info(self):
        return super().info() + [
            self.absolute_discrimination,
            self.discrimination_index,
            self.novelty_preference,
        ]


@dataclass(frozen=True, order=True)
class NortObjectField:
    label: int
    constant_object_perimeter: PolygonalPerimeter
    variable_object_perimeter: PolygonalPerimeter
    novel_object_perimeter: PolygonalPerimeter
    novelty_constant_object_perimeter: Union[PolygonalPerimeter, None] = None

    def __post_init__(self):
        if self.novelty_constant_object_perimeter:
            self.constant_object_perimeter.semantic_label = "habituation_constant"
            self.novelty_constant_object_perimeter.semantic_label = "novelty_constant"
        else:
            self.constant_object_perimeter.semantic_label = "constant"

        self.variable_object_perimeter.semantic_label = "variable"
        self.novel_object_perimeter.semantic_label = "novel"

    @classmethod
    def from_images(cls, label: int, habituation_img: Any, novelty_img: Any):
        polygon_n = 4
        return cls.from_undefined(
            label=int(label),
            habituation_object_perimeter_a=PolygonalPerimeter.from_image(
                habituation_img, n=polygon_n
            ),
            habituation_object_perimeter_b=PolygonalPerimeter.from_image(
                habituation_img, n=polygon_n
            ),
            novel_object_perimeter=PolygonalPerimeter.from_image(
                novelty_img, n=polygon_n, semantic_label="novel"
            ),
        )

    @classmethod
    def from_undefined(
        cls,
        label: int,
        habituation_object_perimeter_a: PolygonalPerimeter,
        habituation_object_perimeter_b: PolygonalPerimeter,
        novel_object_perimeter: PolygonalPerimeter,
        novelty_constant_object_perimeter: Union[PolygonalPerimeter, None] = None,
    ):
        if PolygonalPerimeter.distance_between_two_vectors(
            habituation_object_perimeter_a, novel_object_perimeter
        ) < PolygonalPerimeter.distance_between_two_vectors(
            habituation_object_perimeter_b, novel_object_perimeter
        ):
            constant_object_perimeter = habituation_object_perimeter_b
            variable_object_perimeter = habituation_object_perimeter_a
        else:
            constant_object_perimeter = habituation_object_perimeter_a
            variable_object_perimeter = habituation_object_perimeter_b

        return cls(
            label,
            constant_object_perimeter,
            variable_object_perimeter,
            novel_object_perimeter,
            novelty_constant_object_perimeter,
        )

    def training(self, *args, **kwargs) -> NortTrainingTrial:
        return NortTrainingTrial(
            nort_a=self.constant_object_perimeter,
            nort_b=self.variable_object_perimeter,
            *args,
            **kwargs,
        )

    def novelty(self, *args, **kwargs) -> NortTrainingTrial:
        return NortNoveltyTrial(
            nort_a=self.novelty_constant_object_perimeter
            or self.constant_object_perimeter,
            nort_b=self.novel_object_perimeter,
            *args,
            **kwargs,
        )

    def pickle(self, path: Any):
        name = f"{self.__class__.__name__}_{self.label}"

        i = 1
        while (filepath := Path(path) / (name + ".lz4")).exists():
            name += f"_{(i := 1 + i)}"

        with open(filepath, "wb") as f:
            compress_pickle.dump(self, f)


@dataclass(frozen=True, order=True)
class NortTrainingToNovelty:
    animal_id: int
    training: NortTrainingTrial = field(compare=False)
    novelty: NortNoveltyTrial = field(compare=False)
