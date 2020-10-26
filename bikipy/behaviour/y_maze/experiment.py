from typing import Union, SupportsFloat, Sequence, Iterable, Dict, List, AnyStr
from itertools import permutations

import pandas as pd
import numpy as np

from bikipy.features.movement import displacement_mean_speed_acceleration
from bikipy.behaviour.y_maze.utils import (
    unique_with_counts_zipped,
    exclude_value_from_sequence,
    triplet_permutation_vs_base_permutation_dictionary,
    reduce_location_sequence,
)
from bikipy.border.base import PolygonalBorder


class YMaze:
    def __init__(
        self,
        coordinate_sequence: Sequence[Sequence[SupportsFloat]],
        arms: Sequence[PolygonalBorder],
        center: PolygonalBorder,
        fps: SupportsFloat,
        cm_per_pixel: SupportsFloat,
        label: Union[Sequence[AnyStr], AnyStr, None] = None,
    ):
        """
        Parameters
        ----------
        coordinate_sequence: Sequence
            The coordinates of the subject across the frames of the video recording
        arms: Sequence
            bikipy border objects defining the arms of the y-maze
        center
            bikipy border object defining the centre of the y-maze
        fps: SupportsFloat
            Number of frames per second
        cm_per_pixel: SupportsFloat
            Number defining the number of pixels that goes into one centimeter
        label: Sequence[AnyStr], AnyStr; optional

        """

        self.fps = float(fps)
        self.cm_per_pixel = float(cm_per_pixel)
        self.label = label

        self.arms = arms
        self.center = center
        self.coordinate_sequence = np.asanyarray(coordinate_sequence)
        self.location_sequence = PolygonalBorder.detect_sequential_border_presence(
            self.coordinate_sequence,
            self.arms,
            inferior_poly_border_instances=[self.center],
        )
        self.reduced_sequence = reduce_location_sequence(self.location_sequence)
        self.reduced_without_center = exclude_value_from_sequence(
            self.reduced_sequence, self.center.label
        )

        # Quick computations
        self.sum_of_alternations = len(self.reduced_without_center) - 2
        (
            self.displacement,
            self.mean_speed,
            self.mean_acceleration,
        ) = displacement_mean_speed_acceleration(self.coordinate_sequence, self.fps)

    @property
    def arm_labels(self):
        return "".join([border_object.label for border_object in self.arms])

    @property
    def arm_triplets(self):
        return ["".join(triplet) for triplet in permutations(self.arm_labels)]

    @property
    def _arm_center_label_dict(self):
        return dict.fromkeys((*self.arm_labels, self.center.label))

    @property
    def _arm_triplet_dict(self):
        return {arm: 0 for arm in self.arm_triplets}

    @property
    def triplet_permutation_vs_base_key(self):
        return triplet_permutation_vs_base_permutation_dictionary(self.arm_labels)

    @property
    def seconds_spent_in_areas(self) -> Dict:
        """
        The time spent in each area; arms and center

        Returns
        -------
        Dict, area vs time
        """

        result = self._arm_center_label_dict
        for label, counts in unique_with_counts_zipped(self.location_sequence):
            assert label in result
            result[label] = (counts / self.fps) if self.fps else counts

        return result

    @property
    def arm_alternations(self) -> Dict:
        """
        The number of alternations to every arm and center

        Returns
        -------
        Dict, arm label vs alternations to arm
        """

        result = self._arm_center_label_dict
        for label, counts in unique_with_counts_zipped(self.reduced_sequence):
            assert label in result
            result[label] = counts

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

        return 100.0 * alternations / self.sum_of_alternations

    @staticmethod
    def export_to_dataframe(
        y_maze_objects: Iterable, return_as_dict: bool = False
    ) -> Union[pd.DataFrame, Dict]:
        """
        Export experimental data to pandas DataFrame

        Useful for exporting to files such as hdf, xlsx, csv, etc

        Parameters
        ----------
        y_maze_objects
            Sequence of YMaze objects each depicting an experiment
        return_as_dict
            If True return results as dict

        Returns
        -------
        DataFrame with the combined experiment attributes of all the YMaze objects
        """

        def _generate_key(
            base_key: Union[AnyStr, Sequence], new: Union[AnyStr, Sequence]
        ):
            key = []
            key.append(base_key) if isinstance(base_key, str) else key.extend(base_key)
            key.append(new) if isinstance(new, str) else key.extend(new)
            return tuple(key)

        def _generate_third_order_dict(
            data_dict: Dict, base_key: List, base_feat_label: AnyStr
        ) -> pd.DataFrame:
            return {
                _generate_key(base_key, (base_feat_label, arm)): [time]
                for arm, time in data_dict.items()
            }

        export_data = {}
        for y_maze in y_maze_objects:
            export_data = {
                **export_data,
                _generate_key(y_maze.label, "Displacement"): y_maze.displacement,
                _generate_key(y_maze.label, "Mean speed"): y_maze.mean_speed,
                _generate_key(
                    y_maze.label, "Mean acceleration"
                ): y_maze.mean_acceleration,
                _generate_key(
                    y_maze.label, "Spontaneous alternations"
                ): y_maze.spontaneous_alternations,
                **_generate_third_order_dict(
                    y_maze.seconds_spent_in_areas, y_maze.label, "Seconds in area"
                ),
                **_generate_third_order_dict(
                    y_maze.arm_alternations, y_maze.label, "Arm alternations"
                ),
                **_generate_third_order_dict(
                    y_maze.triplet_alternation_distribution,
                    y_maze.label,
                    "Triplet alternation",
                ),
            }
        return (
            export_data
            if return_as_dict
            else pd.DataFrame.from_dict(export_data, orient="columns")
        )
