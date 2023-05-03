from functools import cached_property
from logging import getLogger
from typing import ClassVar

import numpy as np
from pydantic_numpy.dtype import NDArrayBool, NDArrayUint8

from bikipy.behaviour.utils import reduce_repeating_sequences
from bikipy.core.base import BikipyHashable
from bikipy.core.mixin import FeatureCollectorMixin
from bikipy.core.video import VideoMetadataMixin

logger = getLogger(__name__)


class OnePhysicalObjectSetQualiaAnalysis(BikipyHashable, FeatureCollectorMixin, VideoMetadataMixin):
    analysis_label: str
    physical_object_label_to_observation_boolean_index: dict[str, NDArrayBool]

    overlapping_frame_to_total_frame_warning_ratio: ClassVar[float] = 0.05

    @cached_property
    def _physical_object_label_to_zero(self) -> dict:
        return {label: 0.0 for label in self.physical_object_label_to_observation_boolean_index}

    @property
    def physical_object_labels(self) -> tuple[str, ...]:
        return tuple(self.physical_object_label_to_observation_boolean_index)

    @cached_property
    def physical_object_per_frame(self) -> NDArrayBool:
        return np.logical_or.reduce(
            [
                observation_boolean_index
                for observation_boolean_index in self.physical_object_label_to_observation_boolean_index.values()
            ]
        )

    @cached_property
    def physical_object_total_seconds_observing(self) -> float:
        return np.sum(self.observing_per_frame) / self.video.fps

    @cached_property
    def physical_object_label_to_observation_seconds(self) -> dict[str, int]:
        return {
            label: np.sum(observation_boolean_index) / self.video.fps
            for label, observation_boolean_index in self.physical_object_label_to_observation_boolean_index.items()
        }

    @cached_property
    def physical_object_observation_sequence(self) -> NDArrayUint8:
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
    def physical_object_sum_of_observation_instances(self) -> int:
        return np.sum(
            reduce_repeating_sequences(
                self.observation_sequence != 0, frame_tolerance=self.video.minimum_frames_tolerance
            )
        )
