import os
from functools import cached_property
from logging import getLogger
from typing import Any, ClassVar, Optional

import matplotlib.pyplot as plt
import numpy as np
from pydantic import DirectoryPath, validator

from bikipy.behaviour.utils import reduce_repeating_sequences
from bikipy.core.base_class import BikipyBase
from bikipy.core.typing import NDArrayBool, NDArrayFp64
from bikipy.feature.attention.main import (
    gaze_direction_filter,
    perimeter_attention,
    proximity_filter,
    tolerance_filter,
)
from bikipy.perimeter.base import AnyPerimeter, PerimeterSet

logger = getLogger(__name__)


class PhysicalObject(BikipyBase):
    """
    The physical object is a triadic abstraction of Reader, Perimeter and Trial. This abstraction allows
    us to define methods that require the respective attributes, think of it as a union between the classes!
    """

    perimeter: AnyPerimeter
    reader: Any
    gaze_start_point_label: str
    gaze_travel_direction_point_label: str
    fps: float
    perimeter_border_normal_pixel_magnitude: float
    maximum_radians_inter_gaze_perimeter: float
    minimum_seconds_attention: float
    maximum_seconds_distraction: float
    inspection_dir: Optional[DirectoryPath] = None

    def __len__(self) -> int:
        return self.temporal_resolution

    @property
    def label(self):
        return self.perimeter.best_id

    @cached_property
    def distance_from_per_frame(self) -> NDArrayFp64:
        return np.linalg.norm(self._gaze_travel_direction_point - self.perimeter.centroid, axis=1)

    @cached_property
    def not_observing(self) -> NDArrayFp64:
        return ~self.observance_boolean_index

    @cached_property
    def attention_filtered_seconds_observing(self) -> float:
        return np.sum(self.observance_boolean_index) / self.fps

    @cached_property
    def raw_seconds_observing(self) -> float:
        return np.sum(self.logical_location_and_gaze) / self.fps

    @cached_property
    def filtered_raw_observation_ratio(self) -> float:
        return self.attention_filtered_seconds_observing / self.raw_seconds_observing

    @cached_property
    def attention_proximity_boolean_index(self) -> NDArrayBool:
        return proximity_filter(
            self.perimeter,
            self._gaze_travel_direction_point,
            self.reader[self.gaze_start_point_label],
            self.perimeter_border_normal_pixel_magnitude,
        )

    @cached_property
    def attention_gaze_boolean_index(self) -> NDArrayBool:
        return gaze_direction_filter(
            self.perimeter,
            self._gaze_travel_direction_point,
            self.reader[self.gaze_start_point_label],
            self.maximum_radians_inter_gaze_perimeter,
            # **gaze_filter_kwargs,         # TODO: See perimeter_attention
        )[0]

    @cached_property
    def logical_location_and_gaze(self) -> NDArrayFp64:
        return self.attention_proximity_boolean_index & self.attention_gaze_boolean_index

    @cached_property
    def observance_boolean_index(self) -> NDArrayBool:
        return (
            np.zeros_like(self.logical_location_and_gaze, dtype=bool)
            if np.sum(self.logical_location_and_gaze) < self.fps
            else np.array(
                tolerance_filter(
                    self.logical_location_and_gaze,
                    self.fps,
                    self.minimum_seconds_attention,
                    self.maximum_seconds_distraction,
                )
            )
        )

    @property
    def temporal_resolution(self) -> int:
        return self.reader.frames

    @property
    def _gaze_travel_direction_point(self) -> NDArrayFp64:
        return self.reader[self.gaze_travel_direction_point_label]

    @cached_property
    def _perimeter_attention_data(self) -> tuple:
        return perimeter_attention(
            self.perimeter,
            self._gaze_travel_direction_point,
            self.reader[self.gaze_start_point_label],
            self.fps,
            self.perimeter_border_normal_pixel_magnitude,
            self.maximum_radians_inter_gaze_perimeter,
            self.minimum_seconds_attention,
            self.maximum_seconds_distraction,
        )


class PhysicalObjectSet(BikipyBase):
    """
    The physical object set provides useful methods that compute relational features of physical-objects.
    Some methods are designed specifically for sets with a specific number of objects, while others are general.
    """

    physical_objects: tuple[PhysicalObject, ...]

    overlapping_frame_to_total_frame_warning_ratio: ClassVar[float] = 0.05

    @cached_property
    def __len__(self) -> int:
        return len(self.physical_objects)

    def __getitem__(self, item):
        return self.label_to_physical_object[item]

    @classmethod
    def from_perimeter(cls, *perimeters, **kwargs):
        return cls(physical_objects=tuple(PhysicalObject(perimeter=perimeter, **kwargs) for perimeter in perimeters))

    @classmethod
    def from_perimeter_set(cls, perimeter_set: PerimeterSet):
        assert not perimeter_set.restricted_perimeters
        return cls.from_perimeter(*perimeter_set.perimeters)

    @cached_property
    def frames(self) -> int:
        return len(self._first_object)

    @property
    def fps(self):
        return self._first_object.fps

    @cached_property
    def observing_per_frame(self):
        return np.logical_or.reduce(
            [physical_object.observance_boolean_index for physical_object in self.physical_objects]
        )

    @cached_property
    def not_observing_per_frame(self):
        return ~self.observing_per_frame

    @cached_property
    def seconds_observing(self):
        return np.sum(self.observing_per_frame) / self.fps

    @cached_property
    def seconds_not_observing(self):
        return np.sum(self.not_observing_per_frame) / self.fps

    @cached_property
    def object_specific_observation(self) -> dict[Any, int]:
        return {
            physical_object.label: physical_object.raw_seconds_observing for physical_object in self.physical_objects
        }

    @cached_property
    def observation_sequence(self):
        overlapping_frames = 0

        result = np.zeros(len(self._first_object), dtype=np.uint8)
        for label, physical_object in self.label_to_physical_object.items():
            current_boolean_index = physical_object.observance_boolean_index

            overlapping_frames += np.sum(current_boolean_index & result)

            result[current_boolean_index] = int(label.split("_")[1])  # TODO: Revert

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

        return result

    @cached_property
    def reduced_observation_sequence(self):
        return np.array(reduce_repeating_sequences(self.observation_sequence, frame_tolerance=round(self.fps / 0.35)))

    @cached_property
    def physical_object_id_to_observation_instances(self):
        return {
            label: count
            for label, count in np.dstack(np.unique(self.reduced_observation_sequence, return_counts=True))[0]
        }

    @cached_property
    def sum_of_observation_instances(self):
        return sum(self.physical_object_id_to_observation_instances.values())

    @cached_property
    def object_bias_score(self) -> dict:
        if not self.seconds_observing:
            return self._label_to_zero
        return {
            label: 100.0 * physical_object.attention_filtered_seconds_observing / self.seconds_observing
            for label, physical_object in self.label_to_physical_object.items()
        }

    @cached_property
    def absolute_pair_discrimination(self) -> float:
        if len(self) != 2:
            msg = f"novel constant discrimination requires that the object number of the set is 2, not {len(self)}"
            raise AttributeError(msg)
        return abs(
            self.physical_objects[0].attention_filtered_seconds_observing
            - self.physical_objects[1].attention_filtered_seconds_observing
        )

    @cached_property
    def label_to_physical_object(self) -> dict:
        return {physical_object.label: physical_object for physical_object in self.physical_objects}

    def plot(self, ax: Any = None):
        if not ax:
            fix, ax = plt.subplots()
        for physical_objects in self.physical_objects:
            ax = physical_objects.perimeter.plot(ax=ax)

        return ax

    @cached_property
    def _label_to_zero(self) -> dict:
        return {label: 0.0 for label in self.label_to_physical_object}

    @cached_property
    def _first_object(self):
        return self.physical_objects[0]

    @validator("physical_objects", pre=True)
    def more_than_one_object(cls, value):
        if len(value) <= 1:
            msg = "Number of physical_objects in a set needs to be more than one"
            raise ValueError(msg)
        return value

    @validator("physical_objects", pre=True)
    def identical_temporal_resolution(cls, value):
        if any(len(value[0]) != len(physical_object) for physical_object in value[1:]):
            msg = (
                f"The number of frames differ across physical objects:\n"
                f"{', '.join(str(len(physical_object)) for physical_object in value)}"
            )
            raise AttributeError(msg)
        return value

    @validator("physical_objects", pre=True)
    def identical_fps(cls, value):
        if any(value[0].fps != physical_object.fps for physical_object in value[1:]):
            msg = (
                f"Frames per second differ across physical objects:\n"
                f"{', '.join((physical_object.fps for physical_object in value))}"
            )
            raise AttributeError(msg)
        return value
