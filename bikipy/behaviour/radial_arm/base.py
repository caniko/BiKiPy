import string
from copy import copy
from functools import cached_property, lru_cache
from itertools import chain, permutations
from logging import getLogger
from typing import ClassVar, Optional

import numpy as np
import pandas as pd
from ordered_set import OrderedSet
from pydantic import PositiveInt, validator
from pydantic_numpy.dtype import NDArrayBool

from bikipy.behaviour.core.base import BaseExperiment, BaseTrial
from bikipy.behaviour.utils import (
    feature_2d_multi_indexer,
    reduce_repeating_sequences,
    unique_with_counts_zipped,
)
from bikipy.core.base import BikipyHashable
from bikipy.feature.motion import Motion, bulk_motion_analysis_indexer
from bikipy.perimeter.base import PerimeterSet, SinglePerimeter
from bikipy.perimeter.helper.confinement import (
    ConfinementSequence,
    detect_multi_node_sequential_perimeter_presence,
    inspect_sequential_confinement,
)
from bikipy.perimeter.mixin import TrialWithPerimeterMixin
from bikipy.utils.math.geometry import clockwise_sort_perimeter_centroids

logger = getLogger(__name__)


class RadialMazeBase(BikipyHashable):
    category = "radial_maze"


class BaseRadialMazeExperiment(RadialMazeBase, BaseExperiment):
    pass


class BaseRadialMazeTrial(TrialWithPerimeterMixin, RadialMazeBase, BaseTrial):
    center: SinglePerimeter = ...
    arms: tuple[SinglePerimeter, ...] = ...

    number_of_arms: ClassVar[Optional[int]]

    tracking_labels_for_radial_arm_confinement: OrderedSet[str] = ...

    minimum_seconds_for_entry: float = 0.5

    @classmethod
    @property
    def _arm_int_ids(cls) -> list[PositiveInt]:
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
    def _center_arm_int_ids(cls) -> tuple[PositiveInt, ...]:
        # The center always has int ID 1, and the arms have int IDs starting from 2 in clock-wise order
        return 1, *cls._arm_int_ids

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
    def _center_arm_labels(cls) -> tuple[str, ...]:
        return *cls.arm_labels, "Center"

    @classmethod
    @property
    def schemantic_fields_to_exclude_from_config_schema(cls) -> set[str]:
        result = super().schemantic_fields_to_exclude_from_config_schema
        result.update(("arms", "center"))
        return result

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

    @property
    def perimeters(self):
        """
        Do not change the order of the perimeters, this will break the alternation sequence,
        which uses it as inferior to superior sequence for `detect_multi_node_sequential_perimeter_presence`
        """
        return *self.arms, self.center

    @cached_property
    def arm_len(self):
        return len(self.arms)

    @cached_property
    def perimeter_set(self):
        return PerimeterSet(perimeters=self.perimeters)

    @property
    def alternation_sequence_with_center(self) -> np.ndarray[int, np.dtype[np.uint8]]:
        result, overlap_boolean_index = detect_multi_node_sequential_perimeter_presence(
            self.confinement_coordinates, self.perimeter_set
        )
        inspect_sequential_confinement(
            self.inspect_subdir_or_bool("alternation_sequence_w/center"),
            self.video,
            self.perimeter_set,
            self.reader.kinematic_coordinates,
            result,
            overlap_boolean_index,
        )
        return result

    @cached_property
    def alternation_sequence(self) -> ConfinementSequence:
        perimeter_set_only_arms = PerimeterSet(perimeters=self.arms)
        result, overlap_boolean_index = detect_multi_node_sequential_perimeter_presence(
            self.confinement_coordinates, perimeter_set_only_arms
        )
        inspect_sequential_confinement(
            self.inspect_subdir_or_bool("alternation_sequence"),
            self.video,
            perimeter_set_only_arms,
            self.reader.kinematic_coordinates,
            result,
            overlap_boolean_index,
        )
        return result

    @cached_property
    def reduced_alternation_sequence(self) -> ConfinementSequence:
        result = reduce_repeating_sequences(
            self.alternation_sequence, round(self.video.fps * self.minimum_seconds_for_entry)
        )
        return result[np.nonzero(result)]

    @cached_property
    def sum_of_entries(self) -> int:
        return len(self.reduced_alternation_sequence) - 1

    @cached_property
    def area_to_confinement_boolean_index(self) -> dict[str, NDArrayBool]:
        return {label: self.alternation_sequence_with_center == label for label in self._center_arm_int_ids}

    @property
    def area_to_motion(self) -> dict[str, Motion]:
        return {
            label: Motion(
                coordinate_sequence=self.reader.kinematic_coordinates[confinement_boolean_index], fps=self.video.fps
            ).as_tuple
            for label, confinement_boolean_index in self.area_to_confinement_boolean_index.items()
        }

    @cached_property
    def area_to_seconds_spent(self) -> dict[str, int]:
        return {
            label: self.video.boolean_array_to_seconds(confinement_boolean_index)
            for label, confinement_boolean_index in self.area_to_confinement_boolean_index.items()
        }

    @cached_property
    def sum_of_seconds_in_arms_and_center(self) -> float:
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
        result = {k: int(v) for k, v in unique_with_counts_zipped(self.reduced_alternation_sequence)}

        start_loc = self.reduced_alternation_sequence[0]
        assert result[start_loc] > 0
        result[start_loc] -= 1

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
            current_permutation = self.reduced_alternation_sequence[i : i + self.arm_len]
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
            current_permutation = self.reduced_alternation_sequence[i : i + self.arm_len]
            if all(arm.int_id in current_permutation for arm in self.arms):
                alternations += 1

        alternative_alternations = sum(self.permutation_alternation_distribution.values())
        if alternations != alternative_alternations:
            msg = f"Alternation compute methods yielded differing values: {alternations} != {alternative_alternations}"
            raise ValueError(msg)

        return 100.0 * alternations / (self.sum_of_entries - 2)

    @cached_property
    def confinement_coordinates(self) -> tuple[np.ndarray[float, np.dtype[np.float64]], ...]:
        return tuple(self.reader[node_label] for node_label in self.tracking_labels_for_radial_arm_confinement)

    @cached_property
    def _arm_permutation_to_zero(self):
        return {arm: 0 for arm in self._arm_int_id_permutations}

    @property
    def _arm_center_int_ids(self):
        return self.perimeter_set.perimeter_to_int_id

    @cached_property
    def _arm_center_int_id_to_zero(self):
        return {perimeter.int_id: 0 for perimeter in self._arm_center_int_ids}

    @property
    def _analysis_series_list(self) -> list[pd.Series]:
        upstream_list = super()._analysis_series_list

        # stored in variable for easier debugging

        values = (
            self.spontaneous_alternations,
            *self.arm_to_entries.values(),
            self.sum_of_entries,
            *self.permutation_alternation_distribution.values(),
            self.sum_of_permutation_alternation_distribution,
            *self.area_to_seconds_spent.values(),
            self.sum_of_seconds_in_arms,
            self.sum_of_seconds_in_arms_and_center,
            *chain(*self.area_to_motion.values()),
        )
        indices = [
            ("SpontaneousAlternations", ""),
            *feature_2d_multi_indexer("ArmEntries", self.arm_labels),
            ("ArmEntries", "Sum"),
            *feature_2d_multi_indexer("PermutationAlternation", self._arm_label_permutations_as_string),
            ("PermutationAlternation", "Sum"),
            *feature_2d_multi_indexer("SecondsInArea", self._center_arm_labels),
            ("SecondsInArea", "Arms"),
            ("SecondsInArea", "All"),
            *bulk_motion_analysis_indexer(self._center_arm_labels, 2),
        ]

        upstream_list.append(pd.Series(values, index=indices))

        return upstream_list


@lru_cache
def _compute_meter_per_pixel(corridor_pixel_length: float, corridor_metric_width: float) -> float:
    return corridor_pixel_length / corridor_metric_width
