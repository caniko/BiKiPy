import itertools as it
from copy import copy
from math import ceil
from typing import Any, Dict, Iterable, Sequence, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from bikipy.behaviour.base import BaseExperiment
from bikipy.behaviour.utils import (
    exclude_value_from_sequence,
    reduce_repeating_sequences,
    triplet_permutation_vs_base_permutation_dictionary,
    unique_with_counts_zipped,
)
from bikipy.behaviour.y_maze.utils import mean_intersecting_points_on_borders
from bikipy.border.base import PolygonalBorder


class YMaze(BaseExperiment):
    def __init__(
        self,
        arms: Sequence[PolygonalBorder],
        center: PolygonalBorder,
        average_intersections: bool = True,
        *args,
        **kwargs,
    ):
        """
        Parameters
        ----------
        arms: Sequence
            bikipy border objects defining the arms of the y-maze
        center
            bikipy border object defining the centre of the y-maze
        """

        super().__init__(*args, **kwargs)

        if average_intersections:
            self.arms, self.center = mean_intersecting_points_on_borders(arms, center)
        else:
            self.arms, self.center = arms, center

        self.alternation_sequence, self.valid_indexes, self.valid_boolean_indexes = \
            PolygonalBorder.detect_sequential_border_presence(
                self.location_sequence,
                self.arms,
                inferior_poly_border_instances=[self.center],
            )
        self.invalid_boolean_indexes = np.logical_not(self.valid_boolean_indexes)

        self.reduced_alternation_sequence = reduce_repeating_sequences(
            self.alternation_sequence
        )
        self.reduced_without_center = exclude_value_from_sequence(
            self.reduced_alternation_sequence, self.center.int_label
        )

        self.sum_of_alternations = len(self.reduced_without_center) - 2
        assert self.sum_of_alternations > 0, self.sum_of_alternations

        # Data labeling helpers
        self.arm_labels = [arm.semantic_label for arm in self.arms]
        self.arm_labels_str = "".join(self.arm_labels)

        self.arm_triplets = [
            "".join(triplet) for triplet in it.permutations(self.arm_labels_str)
        ]
        self.arm_center_labels = self.arm_labels + [self.center.semantic_label]

        self._arm_center_label_dict = {area: 0 for area in self.arm_center_labels}
        self._arm_triplet_dict = {arm: 0 for arm in self.arm_triplets}

        self.arm_triplet_combinations = it.combinations_with_replacement(
            self.arm_triplets, 3
        )
        self.triplet_permutation_vs_base_key = (
            triplet_permutation_vs_base_permutation_dictionary(self.arm_labels_str)
        )

    @property
    def seconds_spent_in_areas(self) -> Dict:
        """
        The time spent in each area; arms and center

        Returns
        -------
        Dict, area vs time
        """

        result = copy(self._arm_center_label_dict)
        for label, counts in unique_with_counts_zipped(self.alternation_sequence):
            assert label in result
            result[label] = (counts / self.fps) if self.fps else counts

        return result

    @property
    def area_alternations(self) -> Dict:
        """
        The number of alternations to every arm and center

        Returns
        -------
        Dict, arm label vs alternations to arm
        """

        result = copy(self._arm_center_label_dict)
        for label, counts in unique_with_counts_zipped(
            self.reduced_alternation_sequence
        ):
            assert label in result
            result[label] = counts

        total_arm_alternations = np.sum([result[lab] for lab in self.arm_labels])
        minimum_center_entries = ceil(total_arm_alternations / 2)

        if not result[self.center.semantic_label]:
            result[self.center.semantic_label] = 0

        if result[self.center.semantic_label] < minimum_center_entries:
            print(
                f"{self.semantic_label}: The number of alternations to the center, "
                f"{result[self.center.semantic_label]} can't be less than the "
                f"ceil of half of the total arm alternations, {minimum_center_entries}"
            )

        return result

    @property
    def triplet_alternation_distribution(self) -> Dict:
        """
        Define the triplet alternation distribution.

        A y-maze has three arms and one center, compute the number of occurrences
        a given triplet has. There are six possible triplets, six factorial (6!).

        Returns
        -------
        Dict, triplet vs number of occurrences.
        """
        distribution = self._arm_triplet_dict
        for i in range(self.sum_of_alternations):
            current_triplet = (
                f"{self.reduced_without_center[i]}"
                f"{self.reduced_without_center[i + 1]}"
                f"{self.reduced_without_center[i + 2]}"
            )

            if (
                "A" in current_triplet
                and "B" in current_triplet
                and "C" in current_triplet
            ):
                distribution[current_triplet] += 1

        return distribution

    @property
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

        alternations = 0
        for i in range(self.sum_of_alternations):
            current_triplet = (
                f"{self.reduced_without_center[i]}"
                f"{self.reduced_without_center[i+1]}"
                f"{self.reduced_without_center[i+2]}"
            )
            if (
                "A" in current_triplet
                and "B" in current_triplet
                and "C" in current_triplet
            ):
                alternations += 1

        assert self.sum_of_alternations > 0

        return 100.0 * alternations / self.sum_of_alternations

    def plot(self, ax: Any = None, points: Union[Sequence, None] = None, invalid: bool = False):
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
            points=points or self.location_sequence[
                self.invalid_boolean_indexes if invalid else self.valid_boolean_indexes
            ],
        )

        return ax

    @staticmethod
    def export_to_dataframe(y_maze_objects: Iterable) -> pd.DataFrame:
        """
        Export experimental data to pandas DataFrame

        Useful for exporting to files such as hdf, xlsx, csv, etc

        Parameters
        ----------
        y_maze_objects
            Sequence of YMaze objects each depicting an experiment

        Returns
        -------
        DataFrame with the combined experiment attributes of all the YMaze objects
        """

        def feature_area(feature, arm_center_labels):
            return tuple([(feature, area) for area in arm_center_labels])

        def feature_triplet(feature, triplets):
            return tuple([(feature, area) for area in triplets])

        first = tuple(y_maze_objects)[0]
        feature_order = pd.MultiIndex.from_tuples(
            (
                ("Displacement", ""),
                ("Mean speed", ""),
                ("Mean acceleration", ""),
                ("Spontaneous alternations", ""),
                *feature_area("Seconds in area", first.arm_center_labels),
                *feature_area("Area alternations", first.arm_center_labels),
                *feature_triplet("Triplet alternation", first.arm_triplets),
            ),
            names=("Feature", "Area/Triplet"),
        )
        print(feature_order)

        unit_length = None
        index_vs_data = {}
        for y_maze in y_maze_objects:
            index_vs_data[y_maze.semantic_label] = (
                y_maze.displacement,
                y_maze.mean_speed,
                y_maze.mean_acceleration,
                y_maze.spontaneous_alternations,
                *tuple(y_maze.seconds_spent_in_areas.values()),
                *tuple(y_maze.area_alternations.values()),
                *tuple(y_maze.triplet_alternation_distribution.values()),
            )
            if not unit_length:
                unit_length = len(index_vs_data[y_maze.semantic_label])

        index_vs_data = dict(sorted(index_vs_data.items(), key=lambda item: item[0]))

        return pd.DataFrame(
            tuple(index_vs_data.values()),
            index=tuple(index_vs_data.keys()),
            columns=feature_order,
        )
