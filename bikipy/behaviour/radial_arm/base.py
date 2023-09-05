import string
from functools import cached_property
from itertools import chain, permutations
from logging import getLogger
from typing import ClassVar, Optional

import numpy as np
import pandas as pd
from ordered_set import OrderedSet
from pydantic import DirectoryPath, PositiveInt, computed_field, field_validator
from pydantic_numpy.typing import NpNDArrayBool

from bikipy._constant import INSPECT_SIMPLE_FIG_FILE_FORMAT
from bikipy.analysis.video import make_inspection_video
from bikipy.behaviour.core.base import BaseExperiment, BaseTrial
from bikipy.behaviour.utils import feature_2d_multi_indexer, unique_with_counts_zipped
from bikipy.core.base import BikipyHashable
from bikipy.feature.motion import Motion, bulk_motion_analysis_indexer
from bikipy.math import clockwise_sort_perimeter_centroids
from bikipy.math.discrete import reduce_repeating_sequences
from bikipy.perimeter.base import PerimeterSet, SinglePerimeter
from bikipy.perimeter.trial_mixin import TrialWithPerimeterMixin
from bikipy.perimeter.utils.multi_node_confinement import (
    ConfinementSequence,
    detect_multi_node_sequential_perimeter_presence,
    inspect_sequential_confinement,
)
from bikipy.utils.plot.inspect import generic_inspection_finalization

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

    minimum_seconds_for_entry: float = 0

    @field_validator("center")
    def center_has_1_as_int_id(cls, value):
        value.int_id = 1
        return value

    @field_validator("arms", pre=True)
    def clockwise_sort_and_incremental_arm_int_ids(cls, value):
        value = clockwise_sort_perimeter_centroids(value)
        for i, arm in enumerate(value):
            arm.int_id = cls._arm_int_ids[i]
        return tuple(value)

    @computed_field
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

    @computed_field
    @classmethod
    @property
    def _center_arm_int_ids(cls) -> tuple[PositiveInt, ...]:
        # The center always has int ID 1, and the arms have int IDs starting from 2 in clock-wise order
        return 1, *cls._arm_int_ids

    @computed_field
    @classmethod
    @property
    def arm_labels(cls) -> list:
        return list(string.ascii_uppercase[: cls.number_of_arms])

    @computed_field
    @classmethod
    @property
    def int_ids_to_labels(cls) -> dict[int, str]:
        result = {int_id: label for int_id, label in zip(cls._arm_int_ids, cls.arm_labels)}
        result[1] = "Center"
        return result

    @computed_field
    @classmethod
    @property
    def _arm_int_id_permutations(cls):
        return permutations(cls._arm_int_ids)

    @computed_field
    @classmethod
    @property
    def _arm_label_permutations(cls):
        return permutations(cls.arm_labels)

    @computed_field
    @classmethod
    @property
    def _arm_label_permutations_as_string(cls):
        return map("".join, cls._arm_label_permutations)

    @computed_field
    @classmethod
    @property
    def _center_arm_labels(cls) -> tuple[str, ...]:
        """
        Notice that the order is reversed compared to cls.perimeters, that is because the indices are sorted
        for the data structures mirrored by these labels
        :return:
        """
        return "Center", *cls.arm_labels

    @computed_field(return_type=set[str])
    @classmethod
    @property
    def schemantic_fields_to_exclude_from_config_schema(cls) -> set[str]:
        result = super().schemantic_fields_to_exclude_from_config_schema
        result.update(("arms", "center"))
        return result

    @computed_field
    @property
    def perimeters(self):
        """
        Do not change the order of the perimeters, this will break the alternation sequence,
        which uses it as inferior to superior sequence for `detect_multi_node_sequential_perimeter_presence`
        """
        return self.center, *self.arms

    @computed_field
    @cached_property
    def arm_len(self):
        return len(self.arms)

    @computed_field
    @cached_property
    def perimeter_set(self):
        return PerimeterSet(perimeters=self.perimeters)

    @computed_field
    @cached_property
    def alternation_sequence_with_center(self) -> NpNDArrayUint8:
        result, overlap_boolean_index = detect_multi_node_sequential_perimeter_presence(
            self.confinement_coordinates,
            self.perimeter_set,
            (False, True, True, True),
            tolerance_filter=True,
            tolerance_fps=self.video.fps,
        )

        # TODO: Pydantic v2, put this in model_post_init
        fig, ax = self.video.subplot()
        ax.scatter(*self.reader.kinematic_coordinates_prepared_for_plotting.T)
        self.perimeter_set.plot(ax, coordinates_as_pixels=True)
        generic_inspection_finalization(
            self.inspection_fig_output_path,
            potential_dir="radial_arm",
            potential_label=self.label,
            inspect_fig_file_format=INSPECT_SIMPLE_FIG_FILE_FORMAT,
        )

        inspect_sequential_confinement(
            self.inspection_fig_output_path,
            self.video,
            self.perimeter_set,
            self.reader.kinematic_coordinates_prepared_for_plotting,
            result,
            overlap_boolean_index,
            potential_dir="alternation_sequence_with_center",
            inspect_fig_file_format=INSPECT_SIMPLE_FIG_FILE_FORMAT,
        )
        return result

    @computed_field
    @property
    def cleaned_arm_alternation_sequence(self) -> ConfinementSequence:
        # When it is nowhere it must be on center; we can safely remove undefined instances
        return self.alternation_sequence_with_center[self.alternation_sequence_with_center != 0]

    @computed_field
    @cached_property
    def reduced_arm_center_alternation_sequence(self) -> ConfinementSequence:
        result = np.array(
            reduce_repeating_sequences(
                self.cleaned_arm_alternation_sequence, round(self.video.fps * self.minimum_seconds_for_entry)
            )
        )
        return result[result != self.center.int_id]

    @computed_field
    @cached_property
    def reduced_arm_alternation_sequence(self) -> ConfinementSequence:
        return self.reduced_arm_center_alternation_sequence[
            self.reduced_arm_center_alternation_sequence != self.center.int_id
        ]

    @computed_field
    @cached_property
    def sum_of_entries(self) -> int:
        result = len(self.reduced_arm_alternation_sequence)
        assert sum(self.arm_to_entries.values()) == result
        return result

    @computed_field
    @cached_property
    def area_to_confinement_boolean_index(self) -> dict[str, NpNDArrayBool]:
        return {label: self.alternation_sequence_with_center == label for label in self._center_arm_int_ids}

    @computed_field
    @property
    def area_to_motion(self) -> dict[str, Motion]:
        return {
            label: Motion(
                coordinate_sequence=self.reader.kinematic_coordinates[confinement_boolean_index], fps=self.video.fps
            ).as_tuple
            for label, confinement_boolean_index in self.area_to_confinement_boolean_index.items()
        }

    @computed_field
    @cached_property
    def area_to_seconds_spent(self) -> dict[str, int]:
        return {
            label: self.video.boolean_array_to_seconds(confinement_boolean_index)
            for label, confinement_boolean_index in self.area_to_confinement_boolean_index.items()
        }

    @computed_field
    @property
    def sum_of_seconds_in_arms(self) -> float:
        return sum(self.area_to_seconds_spent[arm_id] for arm_id in self._arm_int_ids)

    @computed_field
    @property
    def sum_of_seconds_in_arms_and_center(self) -> float:
        return sum(self.area_to_seconds_spent.values())

    @computed_field
    @cached_property
    def arm_to_entries(self) -> dict[str, int]:
        """
        The number of alternations to every arm and center

        Returns
        -------
        dict, arm label vs alternations to arm
        """
        result = {arm: 0 for arm in self._arm_int_ids}
        for arm, count in unique_with_counts_zipped(self.reduced_arm_alternation_sequence):
            result[arm] = count
        return result

    @computed_field
    @cached_property
    def permutation_alternation_distribution(self) -> dict[int, int]:
        """
        Define the permutation alternation distribution.

        A y-maze is radial maze with three arms. The function computes the number
        of occurrences each sequential permutation of arm. There are six possible
        permutations, three factorial (3!).

        Returns
        -------
        dict, permutation vs number of occurrences.
        """
        distribution = {arm: 0 for arm in self._arm_int_id_permutations}  # Switch if error
        for i in range(self.sum_of_entries):
            current_permutation = self.reduced_arm_alternation_sequence[i : i + self.arm_len]
            if all(arm.int_id in current_permutation for arm in self.arms):
                distribution[tuple(current_permutation)] += 1

        return distribution

    @computed_field
    @property
    def sum_of_permutation_alternation_distribution(self) -> int:
        result = sum(iter(self.permutation_alternation_distribution.values()))
        assert sum(self.permutation_alternation_distribution.values()) == result
        return result

    @computed_field
    @property
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
            current_permutation = self.reduced_arm_alternation_sequence[i : i + self.arm_len]
            if all(arm.int_id in current_permutation for arm in self.arms):
                alternations += 1

        alternative_alternations = sum(self.permutation_alternation_distribution.values())
        if alternations != alternative_alternations:
            msg = f"Alternation compute methods yielded differing values: {alternations} != {alternative_alternations}"
            raise ValueError(msg)

        return 100.0 * alternations / (self.sum_of_entries - 2)

    @computed_field
    @property
    def confinement_coordinates(self) -> tuple[NpNDArrayFp64, ...]:
        return tuple(self.reader[node_label] for node_label in self.tracking_labels_for_radial_arm_confinement)

    def generate_inspection_video(
        self, output_directory: Optional[DirectoryPath] = None, *, codec: Optional[str] = None, **kwargs
    ) -> None:
        if not self.minimum_seconds_for_entry:
            return make_inspection_video(
                video_frames=self.video.video_read_frames(),
                reader=self.reader,
                perimeter_to_boolean_index={
                    perimeter: self.alternation_sequence_with_center == perimeter.int_id
                    for perimeter in self.perimeters
                },
                output_file_path=self._video_file_name(output_directory, context_label="radial_arm"),
                codec=codec,
            )

        raise NotImplementedError()

    @computed_field
    @property
    def _arm_center_int_ids(self):
        return self.perimeter_set.perimeter_to_int_id

    @computed_field
    @property
    def _analysis_series_list(self) -> list[pd.Series]:
        upstream_list = super()._analysis_series_list

        values = (
            self.spontaneous_alternations,
            *self.arm_to_entries.values(),
            self.sum_of_entries,
            *self.permutation_alternation_distribution.values(),
            self.sum_of_permutation_alternation_distribution,
            *self.area_to_seconds_spent.values(),
            self.sum_of_seconds_in_arms,
            self.sum_of_seconds_in_arms_and_center,
            *chain.from_iterable(self.area_to_motion.values()),
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
