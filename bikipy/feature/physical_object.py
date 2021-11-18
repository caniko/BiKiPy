from functools import cached_property
from pathlib import PurePath
from typing import Any, Union

import numpy as np
from pydantic.dataclasses import dataclass

from bikipy.behaviour.utils import reduce_repeating_sequences
from bikipy.feature.attention import polygonal_perimeter_attention


@dataclass(frozen=True, order=True)
class PhysicalObject:
    reader: Any
    gaze_travel_direction_point_label: str
    gaze_start_point_label: str
    fps: float
    perimeter_border_normal_pixel_magnitude: float
    maximum_radians_inter_gaze_perimeter: float
    minimum_seconds_attention: float
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

    def __post_init__(self):
        if len(self.physical_objects) != 1 and any(
            (
                len(self.physical_objects[0]) != len(physical_object)
                for physical_object in self.physical_objects[1:]
            )
        ):
            temporal_resolutions = (
                str(len(physical_object)) for physical_object in self.physical_objects
            )
            msg = (
                f"PhysicalObject instances in given {self.__class__.__name__} "
                f"needs to have identical temporal resolution. Current state:\n"
                f"{', '.join(temporal_resolutions)}"
            )
            raise AttributeError(msg)

    @cached_property
    def int_id_vs_physical_object(self) -> dict:
        return {
            int_id: physical_object
            for int_id, physical_object in enumerate(self.physical_objects, start=1)
        }

    @cached_property
    def observation_sequence(self):
        result = np.zeros(len(self.physical_objects[0]), dtype=np.uint8)
        for int_id, physical_object in self.int_id_vs_physical_object.items():
            result[physical_object.observance_boolean_index] = int_id
        return result

    @cached_property
    def reduced_observation_sequence(self):
        return np.array(
            reduce_repeating_sequences(
                self.observation_sequence, frame_tolerance=self._frame_tolerance
            )
        )

    @property
    def f(self):
        return ~np.logical_or.reduce((physical_object.observance_per_frame for physical_object in self.physical_objects))

    @cached_property
    def physical_object_id_vs_observation_instances(self):
        return {int_id: count for int_id, count in np.unique(self.reduced_observation_sequence, return_counts=True)}

    @cached_property
    def total_observation_instances(self):
        return sum(self.physical_object_id_vs_observation_instances.values())
