from collections import Counter
from dataclasses import dataclass
from functools import cached_property
from logging import getLogger
from pathlib import PurePath
from typing import Any, Union, Optional

import matplotlib.pyplot as plt
import numpy as np
from pydantic.dataclasses import dataclass as pydantic_dataclass

from bikipy.behaviour.utils import reduce_repeating_sequences
from bikipy.feature.attention import polygonal_perimeter_attention
from bikipy.perimeter.base import Perimeter2D

logger = getLogger(__name__)


@pydantic_dataclass(frozen=True, order=True)
class PhysicalObject:
    perimeter: Perimeter2D
    reader: Any
    gaze_travel_direction_point_label: str
    gaze_start_point_label: str
    fps: float
    perimeter_border_normal_pixel_magnitude: float
    maximum_radians_inter_gaze_perimeter: float
    minimum_seconds_attention: float
    int_id: Optional[int] = None
    label: Optional[str] = None
    inspect: Union[bool, str, PurePath] = False

    @cached_property
    def _perimeter_attention_data(self) -> tuple:
        return polygonal_perimeter_attention(
            self.perimeter,
            self.reader[self.gaze_travel_direction_point_label],
            self.reader[self.gaze_start_point_label],
            self.fps,
            self.perimeter_border_normal_pixel_magnitude,
            self.maximum_radians_inter_gaze_perimeter,
            self._minimum_seconds_attention,
            inspect=self.inspect,
        )

    @property
    def observance_boolean_index(self) -> np.ndarray:
        return self._perimeter_attention_data[0]

    @property
    def not_observing(self) -> np.ndarray:
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

    @property
    def _attention_analytics(self):
        return self._perimeter_attention_data

    @property
    def attention_proximity_boolean_index(self) -> np.ndarray:
        return self._attention_analytics[0]

    @property
    def attention_gaze_boolean_index(self) -> np.ndarray:
        return self._attention_analytics[1]

    @property
    def logical_location_and_gaze(self):
        return self._attention_analytics[2]

    @property
    def semi_true_observations(self):
        return self.logical_location_and_gaze

    @property
    def temporal_resolution(self):
        return self.reader.frames

    def __len__(self):
        return self.temporal_resolution


@dataclass(frozen=True, order=True)
class PhysicalObjectSet:
    physical_objects: tuple
    use_label_as_id: bool = False

    def __post_init__(self):
        if len(self) == 1:
            return

        not_identical_error_base = (
            f"PhysicalObject instances in {self.__class__.__name__} "
            f"needs to have identical"
        )
        if any(
            (
                len(self._first_object) != len(physical_object)
                for physical_object in self.physical_objects[1:]
            )
        ):
            temporal_resolutions = (
                str(len(physical_object)) for physical_object in self.physical_objects
            )
            msg = (
                f"{not_identical_error_base} temporal resolution. Current state:\n"
                f"{', '.join(temporal_resolutions)}"
            )
            raise AttributeError(msg)

        if any(
            self._first_object.fps != physical_object.fps
            for physical_object in self.physical_objects[1:]
        ):
            fps_values = (
                physical_object.fps for physical_object in self.physical_objects
            )
            msg = (
                f"{not_identical_error_base} frame per second (fps). Current state:\n"
                f"{', '.join(fps_values)}"
            )
            raise AttributeError(msg)

        object_int_ids = [
            physical_object.int_id for physical_object in self.physical_objects
        ]
        int_ids_set = set(object_int_ids)

        len_total = len(object_int_ids)
        len_unique = len(int_ids_set)
        if len_total != len_unique:
            msg = (
                "At least two of the int_id values are equal, these values "
                "are mutually exclusive"
            )
            raise AttributeError(msg)
        if None in int_ids_set and len_unique != 1:
            msg = (
                "Either none or all of PhysicalObjects need to have their int_ids"
                "defined"
            )
            raise AttributeError(msg)
        if int_ids_set != set(range(1, len_total + 1)):
            msg = (
                "The int_ids must be incremental. IDs that do not follow this rule "
                "must be stored in the label attribute"
            )
            raise AttributeError(msg)

        object_labels = [
            physical_object.label for physical_object in self.physical_objects
        ]
        label_set = set(object_labels)
        counter = Counter(object_labels)
        if any(counter[value] > 1 for value in label_set if value is not None):
            msg = "labels need to be unique with the exception of None"
            raise AttributeError(msg)

    @cached_property
    def frames(self):
        return len(self._first_object)

    @property
    def fps(self):
        return self._first_object.fps

    @cached_property
    def observing_per_frame(self):
        return np.logical_or.reduce(
            (
                physical_object.observance_boolean_index
                for physical_object in self.physical_objects
            )
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
    def observation_sequence(self):
        overlapping_frames = 0

        result = np.zeros(len(self._first_object), dtype=np.uint8)
        for int_id, physical_object in self.int_id_vs_physical_object.items():
            current_boolean_index = physical_object.observance_boolean_index

            overlapping_frames += np.sum(current_boolean_index & result)

            result[current_boolean_index] = int_id

        if overlapping_frames:
            logger.warning(
                f"{overlapping_frames} out of {self.frames} overlap. This is a "
                "soft warning; do not be alarmed. Two or more objects have a temporal "
                "overlap in their observation_sequence. This is a limitation of "
                "the current methodology (refer to the docs). This warning can be "
                "ignored in most instances.\n"
                "You have been warned that this observation_sequence data "
                "MIGHT be empirically wrong. "
            )

        return result

    @cached_property
    def reduced_observation_sequence(self):
        return np.array(
            reduce_repeating_sequences(
                self.observation_sequence, frame_tolerance=self.fps / 0.35
            )
        )

    @cached_property
    def physical_object_id_vs_observation_instances(self):
        return {
            int_id: count
            for int_id, count in np.unique(
                self.reduced_observation_sequence, return_counts=True
            )
        }

    @cached_property
    def sum_of_observation_instances(self):
        return sum(self.physical_object_id_vs_observation_instances.values())

    @cached_property
    def object_bias_score(self) -> dict:
        if not self.seconds_observing:
            return self._int_id_vs_zero
        return {
            int_id: 100.0 * physical_object / self.seconds_observing
            for int_id, physical_object in self.int_id_vs_physical_object.items()
        }

    @cached_property
    def absolute_discrimination(self) -> float:
        """
        Definition: <frames observing novel object> - <frames observing constant object>

        Note: The first PhysicalObject in physical_objects must be the novel object!

        :return:
        """
        if len(self) != 2:
            msg = (
                f"Absolute discrimination is a feature that is only supported when "
                f"the number of PhysicalObjects in the {self.__class__.__name__} "
                f"is two (2)."
            )
            raise AttributeError(msg)

        return np.sum(self.physical_objects[0].observance_boolean_index) - np.sum(
            self.physical_objects[1].observance_boolean_index
        )

    @cached_property
    def int_id_vs_physical_object(self) -> dict:
        return {
            int_id: physical_object
            for int_id, physical_object in zip(self._int_ids, self.physical_objects)
        }

    @cached_property
    def __len__(self):
        return len(self.physical_objects)

    @cached_property
    def _int_ids(self):
        if self._first_object.int_id:
            return tuple(
                physical_object.int_id for physical_object in self.physical_objects
            )
        return tuple(range(1, len(self) + 1))

    @cached_property
    def _int_id_vs_zero(self) -> dict:
        return {int_id: 0.0 for int_id in self.int_id_vs_physical_object}

    @cached_property
    def _first_object(self):
        return self.physical_objects[0]

    def plot(self, ax: Any = None):
        if not ax:
            fix, ax = plt.subplots()
        for physical_objects in self.physical_objects:
            ax = physical_objects.perimeter.plot(ax=ax)

        return ax
