"""
The physical object set provides useful methods that compute relational features of physical-objects.
Some methods are designed specifically for sets with a specific number of objects, while others are general.
"""

from functools import cached_property, reduce
from logging import getLogger
from typing import Any, ClassVar, Iterable

import numpy as np
import pandas as pd
from pydantic import validator
from pydantic_numpy.dtype import NDArrayBool, NDArrayUint8

from bikipy.behaviour.utils import reduce_repeating_sequences
from bikipy.core.base_class import BaseBikipyHashable
from bikipy.core.video import VideoMetadata, VideoMetadataMixin
from bikipy.feature.physical_object.single import PhysicalObject
from bikipy.feature.tolerance.single import single_node_tolerance_model
from bikipy.perimeter.base import PerimeterSet, SinglePerimeter
from bikipy.reader.base import Reader

logger = getLogger(__name__)


class PhysicalObjectSetAnalysis(BaseBikipyHashable, VideoMetadataMixin):
    analysis_label: str

    physical_object_label_to_observation_boolean_index: dict[str, NDArrayBool]

    overlapping_frame_to_total_frame_warning_ratio: ClassVar[float] = 0.05

    @property
    def observing_per_frame(self) -> NDArrayBool:
        return np.logical_or.reduce(
            [
                observation_boolean_index
                for observation_boolean_index in self.physical_object_label_to_observation_boolean_index.values()
            ]
        )

    @property
    def total_seconds_observing(self) -> float:
        return np.sum(self.observing_per_frame) / self.video.fps

    @property
    def feature_summary(self) -> pd.Series:
        object_bias_score = pd.Series(
            self.object_bias_score.values(),
            index=[["ObjectBiasScore", f"{self.analysis_label}_{label}"] for label in self.labels],
        )
        total_seconds_observing = pd.Series(
            (
                self.total_seconds_observing,
                *(
                    observation_seconds
                    for observation_seconds in self.physical_object_label_to_observation_seconds.values()
                ),
            ),
            index=(
                ("SecondsObserving", f"{self.analysis_label}_Total"),
                *[["SecondsObserving", f"{self.analysis_label}_{label}"] for label in self.labels],
            ),
        )
        return pd.concat((object_bias_score, total_seconds_observing))

    @cached_property
    def labels(self) -> list[str, ...]:
        return list(self.physical_object_label_to_observation_boolean_index)

    @cached_property
    def frames(self):
        return len(
            # Using the first object in dictionary
            tuple(self.physical_object_label_to_observation_boolean_index.values())[0]
        )

    @cached_property
    def physical_object_label_to_observation_seconds(self) -> dict[str, int]:
        return {
            label: np.sum(observation_boolean_index) / self.video.fps
            for label, observation_boolean_index in self.physical_object_label_to_observation_boolean_index.items()
        }

    @cached_property
    def observation_sequence(self) -> NDArrayUint8:
        overlapping_frames = 0

        observation_sequence = np.zeros(self.frames, dtype=np.uint8)
        for object_int_id, object_boolean_index in enumerate(
            self.physical_object_label_to_observation_boolean_index.values(), start=1
        ):
            overlapping_frames += np.sum(object_boolean_index & observation_sequence)
            observation_sequence[object_boolean_index] = object_int_id  # TODO: Revert

        if overlapping_frames:
            ratio = overlapping_frames / self.frames
            logger.info(f"{overlapping_frames} out of {self.frames} overlap; ratio {overlapping_frames / self.frames}")
            if ratio > self.overlapping_frame_to_total_frame_warning_ratio:
                logger.warning(
                    f"Ratio is above the warning ratio, {self.overlapping_frame_to_total_frame_warning_ratio}:\n"
                    "This is a soft warning; do not be alarmed. Two or more objects have a temporal "
                    "overlap in their observation_sequence. This warning can be  ignored in most instances. "
                    "You have been warned that this observation_sequence data "
                    "MIGHT be empirically wrong. You may re-annotate the perimeters "
                    "of the objects for improved results"
                )

        return observation_sequence

    @cached_property
    def sum_of_observation_instances(self) -> int:
        return np.sum(
            reduce_repeating_sequences(
                self.observation_sequence != 0, frame_tolerance=self.video.minimum_frames_tolerance
            )
        )

    @cached_property
    def relative_object_bias_score(self) -> dict[str, float]:
        if not self.total_seconds_observing:
            return self._label_to_zero
        return {
            label: 100.0 * np.sum(observation_boolean_index) / self.total_seconds_observing
            for label, observation_boolean_index in self.physical_object_label_to_observation_boolean_index.items()
        }

    @property
    def object_bias_score(self) -> dict[str, float]:
        """
        Academic literature, and statistics concludes relative_object_bias_score is the best for comparison across
        animals.
        """
        return self.relative_object_bias_score

    @cached_property
    def absolute_object_bias_score(self) -> dict[str, float]:
        if not self.total_seconds_observing:
            return self._label_to_zero
        return {
            label: 100.0 * physical_object.tolerance_modeled_proximity_and_gaze_seconds / (self.frames * self.fps)
            for label, physical_object in self.physical_object_label_to_observation_boolean_index.items()
        }

    @cached_property
    def absolute_pair_discrimination(self) -> float:
        if len(self) != 2:
            msg = f"novel constant discrimination requires that the object number of the set is 2, not {len(self)}"
            raise AttributeError(msg)
        seconds = tuple(self.physical_object_label_to_observation_seconds.values())
        return abs(seconds[0] - seconds[1])

    @cached_property
    def _label_to_zero(self) -> dict:
        return {label: 0.0 for label in self.physical_object_label_to_observation_boolean_index}


class PhysicalObjectSet(VideoMetadataMixin):
    physical_objects: tuple[PhysicalObject, ...] = ...

    @validator("physical_objects", pre=True)
    def more_than_one_object(cls, value):
        if len(value) <= 1:
            msg = "Number of physical_objects in a set needs to be more than one"
            raise ValueError(msg)
        return value

    @validator("physical_objects", pre=True)
    def identical_frames(cls, value):
        if any(len(value[0]) != len(physical_object) for physical_object in value[1:]):
            msg = (
                f"The number of frames differ across physical objects:\n"
                f"{', '.join(str(len(physical_object)) for physical_object in value)}"
            )
            raise AttributeError(msg)
        return value

    @validator("physical_objects", pre=True)
    def identical_fps(cls, value):
        if any(value[0].video.fps != physical_object.video.fps for physical_object in value[1:]):
            msg = (
                f"Frames per second differ across physical objects:\n"
                f"{', '.join((physical_object.video.fps for physical_object in value))}"
            )
            raise AttributeError(msg)
        return value

    # @validator("physical_objects")
    # def readers_must_be_identical(cls, value) -> tuple[PhysicalObject, ...]:
    #     if len(value) == 1:
    #         return value
    #     first_reader = value[0].reader
    #     if any(first_reader != other_reader for other_reader in value[1:]):
    #         msg = "Readers of the physical objects are not identical"
    #         raise AttributeError(msg)
    #     return value

    @cached_property
    def __len__(self) -> int:
        return len(self.physical_objects)

    def __getitem__(self, item):
        return self.label_to_physical_object[item]

    @cached_property
    def _video(self):
        return reduce(VideoMetadata.join, (physical_object.video for physical_object in self.physical_objects))

    @property
    def reader(self) -> Reader:
        return self.physical_objects[0].reader

    @cached_property
    def labels(self) -> tuple[str, ...]:
        return tuple(physical_object.label for physical_object in self.physical_objects)

    @cached_property
    def label_to_physical_object(self) -> dict:
        return {physical_object.label: physical_object for physical_object in self.physical_objects}

    @property
    def analysis_objects(self) -> tuple[PhysicalObjectSetAnalysis, ...]:
        return PhysicalObjectSetAnalysis(
            analysis_label="TolGaze",
            physical_object_label_to_observation_boolean_index={
                physical_object.label: physical_object.attention_observance_boolean_index
                for physical_object in self.physical_objects
            },
            manual_video=self.video,
        ), PhysicalObjectSetAnalysis(
            analysis_label="Proximity",
            physical_object_label_to_observation_boolean_index={
                physical_object.label: physical_object.attention_proximity_boolean_index
                for physical_object in self.physical_objects
            },
            manual_video=self.video,
        )

    @property
    def feature_summary(self) -> pd.Series:
        return pd.concat([analysis_object.feature_summary for analysis_object in self.analysis_objects])

    @property
    def seconds_observing(self) -> float:
        return self.analysis_objects[0].total_seconds_observing

    def plot(self, ax: Any = None):
        if not ax:
            fix, ax = self.video.subplots()
        for physical_objects in self.physical_objects:
            ax = physical_objects.perimeter.plot(ax=ax)

        return ax

    @classmethod
    def from_perimeter(cls, *perimeters, **kwargs):
        return cls(physical_objects=tuple(PhysicalObject(perimeter=perimeter, **kwargs) for perimeter in perimeters))

    @classmethod
    def from_perimeter_set(cls, perimeter_set: PerimeterSet):
        assert not perimeter_set.restricted_perimeters
        return cls.from_perimeter(*perimeter_set.perimeters)

    @classmethod
    def from_bikipy_trial(cls, perimeters: Iterable[SinglePerimeter], trial_class):
        return cls(
            physical_objects=tuple(
                PhysicalObject(perimeter=perimeter, **trial_class.physical_object_keyword_arguments)
                for perimeter in perimeters
            )
        )
