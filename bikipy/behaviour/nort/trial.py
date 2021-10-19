"""
Novel Object Recognition test (NORT) class representing a single test.
These tests can be grouped together to form entire experiments.
"""
from dataclasses import dataclass, field
from logging import getLogger
from pathlib import Path
from typing import Any, Union

import numpy as np
from compress_pickle import compress_pickle

from bikipy.behaviour.square import SquareEnclosedTrial
from bikipy.behaviour.utils import reduce_repeating_sequences
from bikipy.feature.attention import polygonal_perimeter_attention
from bikipy.perimeter.base import PolygonalPerimeter

logger = getLogger(__name__)


class NortHabituationTrial(SquareEnclosedTrial):
    """
    NORT experiment without any objects. The purpose of this test is to generate
    reference data for future NORT experiments.
    """

    trial_sequence_index = 0


class NortOpenField(NortHabituationTrial):
    pass


class NortTrainingTrial(NortHabituationTrial):
    trial_sequence_index = 1

    def __init__(
        self,
        nort_a: PolygonalPerimeter,
        nort_b: PolygonalPerimeter,
        nose_label: str,
        center_eye_label: str,
        torso_label: str,
        perimeter_border_normal_metric_magnitude: float,
        maximum_radians_inter_gaze_perimeter: float = 1 / 4 * np.pi,
        *nort_habituation_args,
        **nort_habituation_kwargs,
    ):
        super().__init__(*nort_habituation_args, **nort_habituation_kwargs)

        self.nort_a, self.nort_b = nort_a, nort_b

        self.torso_label, self.center_eye_label, self.nose_label = (
            str(torso_label),
            str(center_eye_label),
            str(nose_label),
        )
        self.maximum_radians_inter_gaze_perimeter = float(
            maximum_radians_inter_gaze_perimeter
        )
        self.perimeter_border_normal_metric_magnitude = float(
            perimeter_border_normal_metric_magnitude
        )
        self.perimeter_border_normal_pixel_magnitude = (
            self.perimeter_border_normal_metric_magnitude / self.units_per_pixel
        )

        (
            self.a_observance_per_frame,
            self.a_proximity_filtered,
            self.a_gaze_filtered,
        ) = self.nort_observation(self.nort_a)
        (
            self.b_observance_per_frame,
            self.b_proximity_filtered,
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
            reduce_repeating_sequences(
                self.observation_sequence, frame_tolerance=self._frame_tolerance
            )
        )

        self.observation_a = np.sum(self.reduced_observation_sequence == 1)
        self.observation_b = np.sum(self.reduced_observation_sequence == 2)
        self.total_observation = self.observation_a + self.observation_b

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

    @property
    def info(self):
        return super().info + [
            self.observation_a,
            self.observation_b,
            self.total_observation,
            self.seconds_spent_a,
            self.seconds_spent_b,
            self.seconds_observing,
            self.object_bias_score,
        ]

    def nort_observation(self, nort_object):
        eye, nose, torso = self.coordinate_sequence[
            self.center_eye_label, self.nose_label, self.torso_label
        ]
        return polygonal_perimeter_attention(
            nort_object,
            nose,
            eye,
            self.fps,
            self.perimeter_border_normal_pixel_magnitude,
            self.maximum_radians_inter_gaze_perimeter,
            inspect=self.inspection_figure_save,
        )

    def plot(self, ax: Any = None):
        ax = super().plot(ax)

        self.nort_a.plot(ax=ax)
        self.nort_b.plot(ax=ax)

        return ax


class NortNoveltyTrial(NortTrainingTrial):
    trial_sequence_index = 2

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.absolute_discrimination = np.sum(self.b_observance_per_frame) - np.sum(
            self.a_observance_per_frame
        )

        self.discrimination_index = (
            self.absolute_discrimination / self.experiment_seconds
        )

        self.novelty_preference = 100.0 * self.seconds_spent_b / self.experiment_seconds

    @property
    def info(self):
        return super().info + [
            self.absolute_discrimination,
            self.discrimination_index,
            self.novelty_preference,
        ]


@dataclass(frozen=True, order=True)
class NortField:
    label: int
    constant_object_perimeter: PolygonalPerimeter
    variable_object_perimeter: PolygonalPerimeter
    novel_object_perimeter: PolygonalPerimeter
    novelty_constant_object_perimeter: Union[PolygonalPerimeter, None] = field(
        default=None
    )
    inspect_image: Any = field(init=False, compare=False, default=None)

    def __post_init__(self):
        if self.novelty_constant_object_perimeter:
            self.constant_object_perimeter.semantic_label = "training_constant"
            self.novelty_constant_object_perimeter.semantic_label = "novel_constant"
        else:
            self.constant_object_perimeter.semantic_label = "constant"

        self.variable_object_perimeter.semantic_label = "variable"
        self.novel_object_perimeter.semantic_label = "novel"

        if self.inspect_image:
            self.constant_object_perimeter.inspect_image = self.inspect_image
            self.variable_object_perimeter.inspect_image = self.inspect_image
            self.novelty_constant_object_perimeter.inspect_image = self.inspect_image

    @classmethod
    def from_images(cls, label: int, inspect_image: Any, novelty_img: Any):
        polygon_n = 4
        return cls.from_undefined(
            label=int(label),
            habituation_object_perimeter_a=PolygonalPerimeter.from_image(
                inspect_image, n=polygon_n
            ),
            habituation_object_perimeter_b=PolygonalPerimeter.from_image(
                inspect_image, n=polygon_n
            ),
            novel_object_perimeter=PolygonalPerimeter.from_image(
                novelty_img, n=polygon_n
            ),
        )

    @classmethod
    def from_undefined(
        cls,
        label: int,
        habituation_object_perimeter_a: PolygonalPerimeter,
        habituation_object_perimeter_b: PolygonalPerimeter,
        novel_object_perimeter: PolygonalPerimeter,
    ):
        if PolygonalPerimeter.distance_between_two_perimeters(
            habituation_object_perimeter_a, novel_object_perimeter
        ) < PolygonalPerimeter.distance_between_two_perimeters(
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
