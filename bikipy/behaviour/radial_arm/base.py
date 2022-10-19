import string
from copy import copy
from functools import cached_property, lru_cache
from itertools import permutations
from logging import getLogger
from typing import ClassVar, Optional

import pandas as pd
from pydantic import validator
from pydantic_numpy.dtype import NDArrayBool, NDArrayFp64, NDArrayUint8

from bikipy.behaviour.core import BaseExperiment, BaseTrial
from bikipy.behaviour.utils import (
    feature_2d_multi_indexer,
    reduce_repeating_sequences,
    unique_with_counts_zipped,
)
from bikipy.core.base_class import BaseBikipyHashable
from bikipy.perimeter.base import PerimeterSet, SinglePerimeter
from bikipy.perimeter.confinement import detect_multi_node_sequential_perimeter_presence
from bikipy.utils.math.geometry import clockwise_sort_perimeter_centroids

logger = getLogger(__name__)


class RadialMazeBase(BaseBikipyHashable):
    number_of_arms: ClassVar[Optional[int]]

    category = "radial_maze"

    @classmethod
    @property
    def _arm_int_ids(cls):
        try:
            return [i for i in range(2, cls.number_of_arms + 2)]
        except AttributeError as e:
            msg = (
                "Either define the number_of_arms class variable manually, "
                "or utilize a fitting subclass that matches the number of "
                "arms in your experiment."
            )
            raise AttributeError(msg) from e

    @classmethod
    @property
    def arm_labels(cls) -> list:
        return list(string.ascii_uppercase[: cls.number_of_arms])

    @classmethod
    @property
    def int_ids_to_labels(cls) -> dict[int, str]:
        result = {int_id: label for int_id, label in zip(cls._arm_int_ids, cls.arm_labels)}
        result[1] = "Center"
        return result

    @classmethod
    @property
    def _arm_int_id_permutations(cls):
        return permutations(cls._arm_int_ids)

    @classmethod
    @property
    def _arm_label_permutations(cls):
        return permutations(cls.arm_labels)

    @classmethod
    @property
    def _arm_label_permutations_as_string(cls):
        return map("".join, cls._arm_label_permutations)

    @classmethod
    @property
    def _arm_center_labels(cls) -> list:
        return [*cls.arm_labels, "Center"]


class BaseRadialMazeExperiment(BaseExperiment, RadialMazeBase):
    pass


class BaseRadialMazeTrial(BaseTrial, RadialMazeBase):
    center: SinglePerimeter = ...
    arms: tuple[SinglePerimeter, ...] = ...

    object_labels_for_y_maze_confinement_tracking: set[str] = ...

    minimum_seconds_for_entry: float = 0.5

    @classmethod
    @property
    def exclude_from_settings_schema(cls) -> set[str]:
        """
        Some required fields for a class are sometimes highly specific to its respective object. These fields should
        be recorded in this class-property to be excluded by the settings generator function in the ingress module
        :return:
        """
        return super().exclude_from_settings_schema.union({"center", "arms"})

    @validator("center")
    def center_has_1_as_int_id(cls, value):
        value.int_id = 1
        return value

    @validator("arms", pre=True)
    def clockwise_sort_and_incremental_arm_int_ids(cls, value):
        value = clockwise_sort_perimeter_centroids(value)
        for i, arm in enumerate(value):
            arm.int_id = cls._arm_int_ids[i]
        return tuple(value)

    @classmethod
    @property
    def radial_arm_feature_headers(cls) -> list[tuple[str, ...]]:
        return [
            ("SpontaneousAlternations", ""),
            *feature_2d_multi_indexer("SecondsInArea", cls._arm_center_labels),
            ("SecondsInArea", "Arms"),
            ("SecondsInArea", "Sum"),
            *feature_2d_multi_indexer("ArmEntries", cls.arm_labels),
            ("ArmEntries", "Sum"),
            *feature_2d_multi_indexer("PermutationAlternation", cls._arm_label_permutations_as_string),
            ("PermutationAlternation", "Sum"),
        ]

    @property
    def _trial_feature_series_list(self) -> list[pd.Series]:
        upstream_list = super()._trial_feature_series_list

        upstream_list.append(
            pd.Series(
                (
                    self.spontaneous_alternations,
                    *self.area_to_seconds_spent.values(),
                    self.sum_of_seconds_in_arms,
                    self.sum_of_seconds_in_perimeters,
                    *self.arm_to_entries.values(),
                    self.sum_of_entries,
                    *self.permutation_alternation_distribution.values(),
                    self.sum_of_permutation_alternation_distribution,
                ),
                index=self.radial_arm_feature_headers,
            )
        )

        return upstream_list

    @cached_property
    def perimeters(self):
        return [*self.arms, self.center]

    @cached_property
    def arm_len(self):
        return len(self.arms)

    @cached_property
    def perimeter_set(self):
        return PerimeterSet(perimeters=self.perimeters)

    @cached_property
    def meters_per_pixel(self):
        return _compute_meter_per_pixel(self.center.mean_length, self.corridor_meter_width)

    @property
    def alternation_sequence(self) -> NDArrayUint8:
        return self._border_presence_data[0]

    @property
    def valid_indices(self) -> NDArrayBool:
        return self._border_presence_data[1]

    @property
    def valid_boolean_index(self) -> NDArrayBool:
        return self._border_presence_data[2]

    @cached_property
    def reduced_alternation_sequence_without_center(self) -> NDArrayUint8:
        return reduce_repeating_sequences(
            self.alternation_sequence, round(self.video.fps * self.minimum_seconds_for_entry)
        )

    @cached_property
    def sum_of_entries(self) -> int:
        return len(self.reduced_alternation_sequence_without_center) - 2

    @property
    def alternation_sequence_with_center(self):
        return self._border_center_presence_data[0]

    @property
    def valid_indices_with_center(self) -> NDArrayBool:
        return self._border_center_presence_data[1]

    @property
    def valid_boolean_index_with_center(self) -> NDArrayBool:
        return self._border_center_presence_data[2]

    @cached_property
    def reduced_alternation_sequence_with_center(self):
        return reduce_repeating_sequences(
            self.alternation_sequence_with_center, round(self.video.fps * self.minimum_seconds_for_entry)
        )

    @cached_property
    def area_to_seconds_spent(self) -> dict:
        """
        The time spent in each area; arms and center

        Returns
        -------
        dict, area vs time
        """

        result = copy(self._arm_center_int_id_to_zero)
        for label, counts in unique_with_counts_zipped(self.alternation_sequence_with_center):
            assert label in result, f"{label} is not in {tuple(result.keys())})"
            result[label] = counts / self.video.fps

        return dict(sorted(result.items()))

    @cached_property
    def sum_of_seconds_in_perimeters(self) -> float:
        return sum(iter(self.area_to_seconds_spent.values()))

    @cached_property
    def sum_of_seconds_in_arms(self) -> float:
        return sum(self.area_to_seconds_spent[arm_id] for arm_id in self._arm_int_ids)

    @cached_property
    def arm_to_entries(self) -> dict[str, int]:
        """
        The number of alternations to every arm and center

        Returns
        -------
        dict, arm label vs alternations to arm
        """
        result = dict(unique_with_counts_zipped(self.reduced_alternation_sequence_without_center))

        if any(arm not in result for arm in self._arm_int_ids):
            missing = set(result).difference(self.int_ids_to_labels)
            logger.warning(
                f"{self.label}: Are missing some of the arms in the arm_to_entries dataset: "
                f"{', '.join(self.int_ids_to_labels[int(int_id)] for int_id in missing)}"
            )

        # TODO: Add this test back without sacrificing data accuracy
        # if result[self.center.int_id] < (minimum_center_entries := ceil(self.sum_of_entries / 2.0)):
        #     logger.warning(
        #         f"{self.center.int_id}: The number of alternations to the center, "
        #         f"{result[self.center.int_id]} can't be less than the "
        #         f"ceil of half of the total arm alternations, {minimum_center_entries}"
        #     )

        return dict(sorted(result.items()))

    @cached_property
    def permutation_alternation_distribution(self) -> dict:
        """
        Define the permutation alternation distribution.

        A y-maze is radial maze with three arms. The function computes the number
        of occurrences each sequential permutation of arm. There are six possible
        permutations, three factorial (3!).

        Returns
        -------
        dict, permutation vs number of occurrences.
        """
        distribution = copy(self._arm_permutation_to_zero)
        for i in range(self.sum_of_entries):
            current_permutation = self.reduced_alternation_sequence_without_center[i : i + self.arm_len]
            if all(arm.int_id in current_permutation for arm in self.arms):
                distribution[tuple(current_permutation)] += 1

        return distribution

    @cached_property
    def sum_of_permutation_alternation_distribution(self) -> int:
        return sum(iter(self.permutation_alternation_distribution.values()))

    @cached_property
    def spontaneous_alternations(self) -> float:
        """
        Define the number of spontaneous alternations between each arm

        A y-maze is radial maze with three arms. The function computes the number
        of occurrences each sequential permutation of arm. There are six possible
        permutations, three factorial (3!).

        Returns
        -------
        float, defining the percentage ratio between permutation consisting of unique
        arms and sum of all permutation alternations.
        """

        assert self.sum_of_entries > 0, self.sum_of_entries

        alternations = 0
        for i in range(self.sum_of_entries):
            current_permutation = self.reduced_alternation_sequence_without_center[i : i + self.arm_len]
            if all(arm.int_id in current_permutation for arm in self.arms):
                alternations += 1

        alternative_alternations = sum(self.permutation_alternation_distribution.values())
        if alternations != alternative_alternations:
            msg = f"Alternation compute methods yielded differing values: {alternations} != {alternative_alternations}"
            raise ValueError(msg)

        return 100.0 * alternations / self.sum_of_entries

    @cached_property
    def _multi_node_coordinates(self) -> tuple[NDArrayFp64, ...]:
        return tuple(self.reader[node_label] for node_label in self.object_labels_for_y_maze_confinement_tracking)

    @cached_property
    def _border_center_presence_data(self):
        return detect_multi_node_sequential_perimeter_presence(
            self._multi_node_coordinates,
            (self.center, *self.arms),
        )

    @cached_property
    def _border_presence_data(self):
        # alternation_sequence, valid_indices, valid_boolean_index
        return detect_multi_node_sequential_perimeter_presence(
            self._multi_node_coordinates,
            self.arms,
            inspect_arg=self.class_inspect_arg,
            inspect_coords=self.kinematic_coordinates,
        )

    @cached_property
    def _arm_permutation_to_zero(self):
        return {arm: 0 for arm in self._arm_int_id_permutations}

    @property
    def _arm_center_int_ids(self):
        return self.perimeter_set.perimeter_to_int_id

    @cached_property
    def _arm_center_int_id_to_zero(self):
        return {perimeter.int_id: 0 for perimeter in self._arm_center_int_ids}


@lru_cache
def _compute_meter_per_pixel(corridor_pixel_length: float, corridor_metric_width: float) -> float:
    return corridor_pixel_length / corridor_metric_width
