import os
from collections import Counter
from functools import cached_property
from logging import getLogger
from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np
from pydantic import DirectoryPath, validator

from bikipy.behaviour.utils import reduce_repeating_sequences
from bikipy.core.base_class import BikipyBase
from bikipy.core.typing import Perimeter2D
from bikipy.feature.attention.main import perimeter_attention
from bikipy.perimeter.base import PerimeterSet
from bikipy.perimeter.io.general import defer_perimeter_set_from_multi_row_reference

logger = getLogger(__name__)


class PhysicalObject(BikipyBase):
    perimeter: Perimeter2D
    reader: Any
    gaze_start_point_label: str
    gaze_travel_direction_point_label: str
    fps: float
    perimeter_border_normal_pixel_magnitude: float
    maximum_radians_inter_gaze_perimeter: float
    minimum_seconds_attention: float
    maximum_seconds_distraction: float
    int_id: Optional[int] = None
    inspection_dir: Optional[DirectoryPath] = None

    def __len__(self) -> int:
        return self.temporal_resolution

    @property
    def label(self):
        return self.perimeter.label

    @cached_property
    def distance_from_per_frame(self) -> np.ndarray:
        return np.linalg.norm(
            self._gaze_travel_direction_point - self.perimeter.centroid, axis=1
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
    def attention_proximity_boolean_index(self) -> np.ndarray:
        return self._attention_analytics[0]

    @property
    def attention_gaze_boolean_index(self) -> np.ndarray:
        return self._attention_analytics[1]

    @property
    def logical_location_and_gaze(self) -> np.ndarray:
        return self._attention_analytics[2]

    @property
    def semi_true_observations(self) -> np.ndarray:
        return self.logical_location_and_gaze

    @property
    def temporal_resolution(self) -> int:
        return self.reader.frames

    @property
    def _gaze_travel_direction_point(self) -> np.ndarray:
        return self.reader[self.gaze_travel_direction_point_label]

    @cached_property
    def _perimeter_attention_data(self) -> tuple:
        if self.inspection_dir:
            if not (perimeter_dir := self.inspection_dir / f"PhyObj_attention_perimeter-{self.perimeter.best_id}").exists():
                os.mkdir(perimeter_dir)
        return perimeter_attention(
            self.perimeter,
            self._gaze_travel_direction_point,
            self.reader[self.gaze_start_point_label],
            self.fps,
            self.perimeter_border_normal_pixel_magnitude,
            self.maximum_radians_inter_gaze_perimeter,
            self.minimum_seconds_attention,
            self.maximum_seconds_distraction,
            inspect=perimeter_dir / f"id_{self.int_id}.jpg" if self.inspection_dir else None
        )

    @property
    def _attention_analytics(self):
        return self._perimeter_attention_data


class PhysicalObjectSet(BikipyBase):
    physical_objects: tuple[PhysicalObject, ...]

    @classmethod
    def from_perimeter(cls, *perimeters, **kwargs):
        return cls(
            physical_objects=tuple(
                PhysicalObject(perimeter=perimeter, int_id=i, **kwargs)
                for i, perimeter in enumerate(perimeters, start=1)
            )
        )

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
            [
                physical_object.observance_boolean_index
                for physical_object in self.physical_objects
            ]
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
        for int_id, physical_object in self._int_id_vs_physical_object.items():
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
            int_id: 100.0
            * physical_object.attention_filtered_seconds_observing
            / self.seconds_observing
            for int_id, physical_object in self._int_id_vs_physical_object.items()
        }

    @cached_property
    def absolute_discrimination(self) -> float:
        """
        Definition: <frames observing novel object> - <frames observing constant object>

        :return:
        """
        if len(self.physical_objects) != 2:
            msg = (
                f"Absolute discrimination is a feature that is only supported when "
                f"the number of PhysicalObjects in the {self.__class__.__name__} "
                f"is two (2)."
            )
            raise AttributeError(msg)

        try:
            return np.sum(
                self._label_vs_physical_object["novel"].observance_boolean_index
            ) - np.sum(
                self._label_vs_physical_object["constant"].observance_boolean_index
            )
        except KeyError:
            msg = (
                "The physical_objects must have a novel and a constant label "
                "to compute absolute_discrimination"
            )
            raise AttributeError(msg)

    @cached_property
    def _int_id_vs_physical_object(self) -> dict:
        return {
            int_id: physical_object
            for int_id, physical_object in zip(self._int_ids, self.physical_objects)
        }

    @cached_property
    def _label_vs_physical_object(self) -> dict:
        return {
            physical_object.label.lower(): physical_object
            for physical_object in self.physical_objects
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
        return {int_id: 0.0 for int_id in self._int_id_vs_physical_object}

    @cached_property
    def _first_object(self):
        return self.physical_objects[0]

    def plot(self, ax: Any = None):
        if not ax:
            fix, ax = plt.subplots()
        for physical_objects in self.physical_objects:
            ax = physical_objects.perimeter.plot(ax=ax)

        return ax

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

    @validator("physical_objects", pre=True)
    def ids_are_unique(cls, value):
        object_int_ids = (physical_object.int_id for physical_object in value)

        int_ids_set = set(object_int_ids)
        len_unique = len(int_ids_set)

        len_total = len(value)

        if len_total != len_unique:
            msg = (
                f"At least two of the int_id values are equal, these int_ids are "
                f"mutually exclusive in {cls.__class__.__name__}:\n"
                f"{', '.join(object_int_ids)}"
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
                "int_ids must be incremental. IDs that do not follow this rule "
                "must be stored in the label attribute"
            )
            raise AttributeError(msg)

        object_labels = (physical_object.label for physical_object in value)
        label_set = set(object_labels)
        counter = Counter(object_labels)
        if any(counter[value] > 1 for value in label_set if value is not None):
            msg = "labels need to be unique with the exception of None"
            raise AttributeError(msg)

        return value


def defer_physical_object_set_from_multi_row_reference(**kwargs):
    """
    Shares the same kwargs as defer_perimeter_set_from_multi_row_reference

    :param kwargs:
    :return:
    """
    image_name_to_perimeter_set = defer_perimeter_set_from_multi_row_reference(**kwargs)
