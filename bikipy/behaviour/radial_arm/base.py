import string
from copy import copy
from functools import cached_property, lru_cache
from itertools import permutations
from logging import getLogger
from typing import ClassVar, Optional

from matplotlib import pyplot as plt
from pydantic import validator

from bikipy.behaviour.base import BaseExperiment, BaseTrial
from bikipy.behaviour.utils import (
    feature_2d_multi_indexer,
    reduce_repeating_sequences,
    unique_with_counts_zipped,
)
from bikipy.core.base_class import BaseBikipyHashable
from bikipy.core.typing import NDArrayBool
from bikipy.perimeter.base import SinglePerimeter, PerimeterSet
from bikipy.perimeter.utils import detect_sequential_border_presence
from bikipy.utils.math.geometry import clockwise_sort_perimeter_centroids

logger = getLogger(__name__)


class RadialMazeBase(BaseBikipyHashable):
    number_of_arms: ClassVar[Optional[int]]

    _class_inspect_directory_name = "RadialMaze"

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
    def _arm_int_id_permutations(cls):
        return permutations(cls._arm_int_ids)

    @classmethod
    @property
    def arm_labels(cls) -> list:
        return list(string.ascii_uppercase[: cls.number_of_arms])

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

    minimum_seconds_for_entry: float = 0.5

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
    def feature_headers(cls) -> list[tuple[str, ...]]:
        return [
            ("SpontaneousAlternations", ""),
            *feature_2d_multi_indexer("SecondsInArea", cls._arm_center_labels),
            ("SecondsInArms", ""),
            ("SumOfSecondsInArea", ""),
            *feature_2d_multi_indexer("ArmEntries", cls.arm_labels),
            ("SumOfEntries", ""),
            *feature_2d_multi_indexer("PermutationAlternation", cls._arm_label_permutations_as_string),
            ("SumOfAlternations", ""),
        ]

    @property
    def feature_df_rows(self) -> list:
        return [
            self.spontaneous_alternations,
            *self.perimeter_to_seconds_spent.values(),
            self.sum_of_seconds_in_arms,
            self.sum_of_seconds_in_perimeters,
            *self.arm_to_entries.values(),
            self.sum_of_entries,
            *self.permutation_alternation_distribution.values(),
            self.sum_of_permutation_alternation_distribution,
        ]

    @cached_property
    def perimeters(self):
        return [*self.sorted_arms, self.center]

    @cached_property
    def arm_len(self):
        return len(self.arms)

    @property
    def sorted_arms(self):
        return clockwise_sort_perimeter_centroids(self.arms)

    @cached_property
    def perimeter_set(self):
        return PerimeterSet(perimeters=self.perimeters)

    @cached_property
    def meters_per_pixel(self):
        return _compute_meter_per_pixel(self.center.mean_length, self.corridor_meter_width)

    @property
    def alternation_sequence(self):
        return self._border_presence_data[0]

    @property
    def valid_indices(self) -> NDArrayBool:
        return self._border_presence_data[1]

    @property
    def valid_boolean_index(self) -> NDArrayBool:
        return self._border_presence_data[2]

    @cached_property
    def reduced_alternation_sequence_without_center(self):
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
    def perimeter_to_seconds_spent(self) -> dict:
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
        return sum(iter(self.perimeter_to_seconds_spent.values()))

    @cached_property
    def sum_of_seconds_in_arms(self) -> float:
        return sum(self.perimeter_to_seconds_spent[arm_id] for arm_id in self._arm_int_ids)

    @cached_property
    def arm_to_entries(self) -> dict[str, int]:
        """
        The number of alternations to every arm and center

        Returns
        -------
        dict, arm label vs alternations to arm
        """
        result = dict(unique_with_counts_zipped(self.reduced_alternation_sequence_without_center))

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
    def _border_center_presence_data(self):
        return detect_sequential_border_presence(
            self.framewise_confined_coordinates,
            self.arms,
            inferior_poly_border_instances=[self.center],
        )

    @cached_property
    def _border_presence_data(self):
        result = detect_sequential_border_presence(
            self.framewise_confined_coordinates,
            self.arms,
        )

        if self.inspect_higher_order:
            fig, ax = plt.subplots()
            ax = self.perimeter_set.plot(coordinates=self.framewise_confined_coordinates, manual_ax=ax)

            if self.inspect_directory:
                plt.savefig(self.class_inspect_directory / f"{self.label}.jpg")
                logger.debug(f"Saved perimeter_set {self.label} inspect plot to {self.class_inspect_directory}")
            else:
                plt.show()

        return result

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
