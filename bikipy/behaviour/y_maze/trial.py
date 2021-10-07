import itertools as it
from copy import copy
from functools import cached_property, partial
from logging import getLogger
from math import ceil
from typing import Any, Sequence, Union

import matplotlib.pyplot as plt
import numpy as np

from bikipy.behaviour.base import BaseTrial
from bikipy.behaviour.utils import (
    exclude_value_from_sequence,
    reduce_repeating_sequences,
    unique_with_counts_zipped,
)
from bikipy.perimeter.base import PolygonalPerimeter
from bikipy.utils.store import translate_keys

INT_TO_SEMANTIC_LABELS = {1: "A", 2: "B", 3: "C", 4: "X"}
generic_int_to_semantic_key_translator = partial(
    translate_keys, translation=INT_TO_SEMANTIC_LABELS
)


logger = getLogger(__name__)


class YMazeTrial(BaseTrial):
    def __init__(
        self,
        arms: Sequence[PolygonalPerimeter],
        center: PolygonalPerimeter,
        *args,
        **kwargs,
    ):
        """
        Parameters
        ----------
        arms: Sequence
            bikipy perimeter objects defining the arms of the y-maze
        center
            bikipy perimeter object defining the centre of the y-maze
        """

        super().__init__(*args, **kwargs)

        self.arms, self.center = arms, center

        (
            self.alternation_sequence,
            self.valid_indices,
            self.valid_boolean_index,
        ) = PolygonalPerimeter.detect_sequential_border_presence(
            self.coordinates_per_frame,
            self.arms,
            inferior_poly_border_instances=[self.center],
        )

        self.invalid_boolean_index = ~self.valid_boolean_index

        self.reduced_alternation_sequence = reduce_repeating_sequences(
            self.alternation_sequence, round(self.fps / 3.0)
        )
        self.reduced_without_center = exclude_value_from_sequence(
            self.reduced_alternation_sequence, self.center.int_label
        )

        self.sum_of_alternations = len(self.reduced_without_center) - 2

        self.arm_int_triplets = [
            triplet for triplet in it.permutations(self.arm_int_labels)
        ]
        self.arm_semantic_triplets = [
            triplet for triplet in it.permutations(self.arm_semantic_labels)
        ]

        self._arm_triplet_dict = {arm: 0 for arm in self.arm_int_triplets}
        self._arm_center_int_label_to_seconds = {
            area: 0 for area in self.arm_center_int_labels
        }

    @property
    def _hash_key(self):
        return self.reduced_without_center

    @cached_property
    def arm_int_labels(self):
        return [arm.int_label for arm in self.arms]

    @cached_property
    def arm_int_labels_array(self):
        return np.array(self.arm_int_labels)

    @cached_property
    def arm_center_int_labels(self):
        return self.arm_int_labels + [self.center.int_label]

    @cached_property
    def arm_semantic_labels(self):
        return [arm.semantic_label for arm in self.arms]

    @cached_property
    def arm_center_semantic_labels(self):
        return self.arm_semantic_labels + [self.center.semantic_label]

    @cached_property
    def int_to_semantic_labels(self):
        return {
            int_label: semantic_label
            for int_label, semantic_label in zip(
                self.arm_center_int_labels, self.arm_center_semantic_labels
            )
        }

    @cached_property
    def seconds_spent_in_areas(self) -> dict:
        """
        The time spent in each area; arms and center

        Returns
        -------
        dict, area vs time
        """

        result = copy(self._arm_center_int_label_to_seconds)
        for label, counts in unique_with_counts_zipped(self.alternation_sequence):
            assert label in result, f"{label} is not in {tuple(result.keys())})"
            result[label] = (counts / self.fps) if self.fps else counts

        return generic_int_to_semantic_key_translator(result)

    @cached_property
    def area_alternations(self) -> dict:
        """
        The number of alternations to every arm and center

        Returns
        -------
        dict, arm label vs alternations to arm
        """

        result = copy(self._arm_center_int_label_to_seconds)
        for label, counts in unique_with_counts_zipped(
            self.reduced_alternation_sequence
        ):
            assert label in result
            result[label] = counts

        if not result[self.center.int_label]:
            result[self.center.int_label] = 0

        if result[self.center.int_label] < (
            minimum_center_entries := ceil(self.sum_of_alternations / 2.0)
        ):
            logger.warning(
                f"{self.center.int_label}: The number of alternations to the center, "
                f"{result[self.center.int_label]} can't be less than the "
                f"ceil of half of the total arm alternations, {minimum_center_entries}"
            )

        return result

    @cached_property
    def triplet_alternation_distribution(self) -> dict:
        """
        Define the triplet alternation distribution.

        A y-maze has three arms and one center, compute the number of occurrences
        a given triplet has. There are six possible triplets, six factorial (6!).

        Returns
        -------
        dict, triplet vs number of occurrences.
        """
        distribution = copy(self._arm_triplet_dict)
        for i in range(self.sum_of_alternations):
            current_triplet = self.reduced_without_center[i : i + 3]
            if 1 in current_triplet and 2 in current_triplet and 3 in current_triplet:
                distribution[tuple(current_triplet)] += 1

        result = {}
        for key, value in distribution.items():
            semantic_key = "".join(
                [self.int_to_semantic_labels[integer] for integer in key]
            )
            result[semantic_key] = value

        return result

    @cached_property
    def spontaneous_alternations(self) -> float:
        """
        Define the number of spontaneous alternations between each y-maze arm

        A y-maze has three arms and one center, compute the number of occurrences
        triplet with unique arms. There are six possible triplets, six factorial (6!).

        Returns
        -------
        float, defining the percentage ratio between triplet consisting of unique arms
        and sum of all triplet alternations.
        """

        if self.sum_of_alternations == 0:
            return 0

        alternations = 0
        for i in range(self.sum_of_alternations):
            current_triplet = self.reduced_without_center[i : i + 3]
            if 1 in current_triplet and 2 in current_triplet and 3 in current_triplet:
                alternations += 1

        assert self.sum_of_alternations > 0, self.sum_of_alternations

        return 100.0 * alternations / self.sum_of_alternations

    def plot(
        self,
        ax: Any = None,
        points: Union[Sequence, None] = None,
        invalid: bool = False,
    ):
        if points and invalid:
            msg = "points can not be defined while invalid is True"
            raise ValueError(msg)

        if not ax:
            fig, ax = plt.subplots()

        for arm in self.arms:
            arm.plot(ax=ax)

        self.center.plot(
            include_borders=False,
            ax=ax,
            bin=True,
            points=(
                points
                or self.coordinates_per_frame[
                    self.invalid_boolean_index if invalid else self.valid_boolean_index
                ]
            ),
        )

        return ax
