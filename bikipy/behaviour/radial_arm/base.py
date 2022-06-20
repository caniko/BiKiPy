import string
from copy import copy
from functools import cached_property, lru_cache
from itertools import permutations
from logging import getLogger
from math import ceil
from typing import ClassVar, Optional

from pydantic import validator

from bikipy.behaviour.base import BaseExperiment, BaseTrial
from bikipy.behaviour.utils import (
    exclude_value_from_sequence,
    feature_2d_multi_indexer,
    reduce_repeating_sequences,
    unique_with_counts_zipped,
)
from bikipy.core.base_class import BikipyBaseHashable
from bikipy.core.typing import NDArrayBool, NDArrayFp64
from bikipy.perimeter.base import AnyPerimeter, PerimeterSet
from bikipy.perimeter.polygon.base import PolygonPerimeter
from bikipy.utils.math.geometry import clockwise_sort_perimeter_centroids

logger = getLogger(__name__)


class RadialMazeBase(BikipyBaseHashable):
    number_of_arms: ClassVar[Optional[int]]

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
    def _arm_labels(cls):
        return string.ascii_uppercase[: cls.number_of_arms]

    @classmethod
    @property
    def _arm_label_permutations(cls):
        return permutations(cls._arm_labels)

    @classmethod
    @property
    def _arm_label_permutations_as_string(cls):
        return map("".join, cls._arm_label_permutations)


class BaseRadialMazeExperiment(BaseExperiment, RadialMazeBase):
    pass


class BaseRadialMazeTrial(BaseTrial, RadialMazeBase):
    corridor_meter_width: float

    center: AnyPerimeter
    arms: tuple[AnyPerimeter, ...]

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
        area_designations = ["Center", *cls._arm_labels]
        return [
            ("Alternations", ""),
            ("Spontaneous alternations", ""),
            *feature_2d_multi_indexer("SecondsInArea", area_designations),
            *feature_2d_multi_indexer("AreaAlternations", area_designations),
            *feature_2d_multi_indexer("PermutationAlternation", cls._arm_label_permutations_as_string),
        ]

    @property
    def feature_df_rows(self) -> list:
        return [
            self.sum_of_alternations,
            self.spontaneous_alternations,
            *self.perimeter_to_seconds_spent.values(),
            *self.perimeter_alternations.values(),
            *self.permutation_alternation_distribution.values(),
        ]

    @cached_property
    def arm_len(self):
        return len(self.arms)

    @property
    def sorted_arms(self):
        return clockwise_sort_perimeter_centroids(self.arms)

    @cached_property
    def perimeter_set(self):
        return PerimeterSet(perimeters=(self.center, *self.sorted_arms))

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
    def invalid_boolean_index(self) -> NDArrayBool:
        return ~self.valid_boolean_index

    @cached_property
    def reduced_alternation_sequence(self):
        return reduce_repeating_sequences(self.alternation_sequence, round(self.fps * 0.075))

    @cached_property
    def reduced_without_center(self):
        return exclude_value_from_sequence(self.reduced_alternation_sequence, self.center.int_id)

    @cached_property
    def sum_of_alternations(self):
        return len(self.reduced_without_center) - 2

    @cached_property
    def perimeter_to_seconds_spent(self) -> dict:
        """
        The time spent in each area; arms and center

        Returns
        -------
        dict, area vs time
        """

        result = copy(self._arm_center_int_id_to_zero)
        for label, counts in unique_with_counts_zipped(self.alternation_sequence):
            assert label in result, f"{label} is not in {tuple(result.keys())})"
            result[label] = (counts / self.fps) if self.fps else counts

        return result

    @cached_property
    def perimeter_alternations(self) -> dict:
        """
        The number of alternations to every arm and center

        Returns
        -------
        dict, arm label vs alternations to arm
        """
        result = dict(unique_with_counts_zipped(self.reduced_alternation_sequence))

        if result[self.center.int_id] < (minimum_center_entries := ceil(self.sum_of_alternations / 2.0)):
            logger.warning(
                f"{self.center.int_id}: The number of alternations to the center, "
                f"{result[self.center.int_id]} can't be less than the "
                f"ceil of half of the total arm alternations, {minimum_center_entries}"
            )

        return result

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
        for i in range(self.sum_of_alternations):
            current_permutation = self.reduced_without_center[i : i + self.arm_len]
            if all(arm.int_id in current_permutation for arm in self.arms):
                distribution[tuple(current_permutation)] += 1

        # result = {}
        # for key, value in distribution.items():
        #     semantic_key = "".join(
        #         [self.perimeter_set.int_id_to_label[integer] for integer in key]
        #     )
        #     result[semantic_key] = value

        return distribution

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

        if self.sum_of_alternations == 0:
            return 0

        alternations = 0
        for i in range(self.sum_of_alternations):
            current_permutation = self.reduced_without_center[i : i + self.arm_len]
            if all(arm.int_id in current_permutation for arm in self.arms):
                alternations += 1

        assert self.sum_of_alternations > 0, self.sum_of_alternations

        return 100.0 * alternations / self.sum_of_alternations

    @classmethod
    def with_reference_point(cls, center: AnyPerimeter, arms: tuple, reference_point: NDArrayFp64, **kwargs):
        return cls(
            center=center.change_reference(reference_point),
            arms=[arm.change_reference(reference_point) for arm in arms],
            **kwargs,
        )

    @cached_property
    def _border_presence_data(self):
        return PolygonPerimeter.detect_sequential_border_presence(
            self.framewise_confined_coordinates,
            self.arms,
            inferior_poly_border_instances=[self.center],
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
