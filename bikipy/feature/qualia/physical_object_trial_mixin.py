from abc import ABC, abstractmethod
from functools import cached_property, partial
from itertools import chain
from typing import Optional, Self

import numpy as np
import pandas as pd
from pydantic import DirectoryPath, Field, computed_field

from bikipy._constant import PHYSICAL_OBJECT_MAP_NAME
from bikipy.analysis.video import make_inspection_video
from bikipy.core.typing import ConfinementSequence
from bikipy.feature.qualia.analysis.mapping import (
    PO_NUMBER_TO_ANALYSIS_MODEL,
)
from bikipy.feature.qualia.heuristic.abc import (
    AbstractHeuristic,
    CombinedHeuristic,
    StandaloneHeuristic,
)
from bikipy.feature.qualia.heuristic.mapping import (
    alias_to_helper_heuristic,
    alias_to_heuristics_cls,
)
from bikipy.feature.qualia.heuristic.mixin import (
    ProximityMixin,
    RayMixin,
)
from bikipy.feature.qualia.heuristic.solo.abc import (
    AbstractSoloHeuristic,
)
from bikipy.feature.qualia.heuristic.merge_parser import (
    parse_heuristic_merge_equation,
)
from bikipy.math.discrete import reduce_repeating_sequences
from bikipy.perimeter.base import BaseSinglePerimeter
from bikipy.perimeter.trial_mixin import TrialWithPerimeterMixin


class PhysicalObjectTrialMixin(TrialWithPerimeterMixin, ABC):
    alias_to_heuristics_combination_equations: dict[str, str] = Field(
        default_factory=dict,
        description="Performs analysis by combining qualia heuristic result with respect to "
        "the defined logical method AND/OR using & or | respectively.",
    )

    @property
    @abstractmethod
    def physical_object_perimeters(self) -> tuple[BaseSinglePerimeter, ...]: ...

    @computed_field  # type: ignore[misc]
    @cached_property
    def perimeters(self) -> tuple[BaseSinglePerimeter, ...]:
        # Inherit and append non-physical-object perimeters to the list
        return tuple(self.physical_object_perimeters)

    @computed_field  # type: ignore[misc]
    @cached_property
    def alias_to_standalone_heuristic(self) -> dict[str, list[StandaloneHeuristic]]:
        result: dict[str, list[AbstractSoloHeuristic]] = {}

        partial_helper_heuristics = []
        alias_to_solo_heuristics = {}
        for heuristic_alias, heuristic_config in self.project_kit_config[PHYSICAL_OBJECT_MAP_NAME].items():
            partial_heuristic_class = partial(
                alias_to_heuristics_cls[heuristic_alias],
                reader=self.reader,
                manual_video=self.video,
                **heuristic_config,
            )
            if heuristic_alias in alias_to_helper_heuristic:
                partial_helper_heuristics.append(partial_heuristic_class)
            else:
                alias_to_solo_heuristics[heuristic_alias] = partial_heuristic_class

        if partial_helper_heuristics:
            perimeter_sequenced_reduced_helper_heuristics = [
                np.logical_or.reduce(
                    [
                        partial_heuristic_class(perimeter=perimeter)
                        for partial_heuristic_class in partial_helper_heuristics
                    ]
                )
                for perimeter in self.perimeters
            ]
        else:
            perimeter_sequenced_reduced_helper_heuristics = [() for _perimeter in self.perimeters]

        for heuristic_alias, partial_heuristic_class in alias_to_solo_heuristics.items():
            result[heuristic_alias] = [
                partial_heuristic_class(perimeter=perimeter, combined_helper_heuristic=reduced_helper_heuristics)
                for perimeter, reduced_helper_heuristics in zip(
                    self.physical_object_perimeters, perimeter_sequenced_reduced_helper_heuristics
                )
            ]

        if self.is_inspecting:
            for heuristic_alias, physical_objects_heuristic in result.items():
                for physical_object_heuristic in physical_objects_heuristic:
                    fig = physical_object_heuristic.plot()
                    self.save_fig(
                        heuristic_alias,
                        base_filename=f"{self.label}_{physical_object_heuristic.physical_object_label}",
                        fig=fig,
                    )

        return result

    @computed_field  # type: ignore[misc]
    @cached_property
    def alias_to_combined_heuristic(self) -> dict[str, list[CombinedHeuristic]]:
        solo_heuristic_alias_to_results = {
            alias: [heuristic.result for heuristic in pos_heuristic]
            for alias, pos_heuristic in self.alias_to_standalone_heuristic.items()
        }

        result = {}
        for heuristic_alias, heuristic_equation in self.alias_to_heuristics_combination_equations.items():
            heuristic_results = parse_heuristic_merge_equation(heuristic_equation, solo_heuristic_alias_to_results)
            result[heuristic_alias] = [
                CombinedHeuristic(
                    perimeter=perimeter,
                    reader=self.reader,
                    label=heuristic_alias,
                    result=heuristic_result,
                    manual_video=self.video,
                )
                for perimeter, heuristic_result in zip(self.physical_object_perimeters, heuristic_results)
            ]

        return result

    @computed_field  # type: ignore[misc]
    @property
    def alias_to_heuristic(self) -> dict[str, AbstractHeuristic]:
        return chain(self.alias_to_standalone_heuristic.items(), self.alias_to_combined_heuristic.items())

    @computed_field  # type: ignore[misc]
    @property
    def physical_object_analysers(self) -> dict[str, Self]:
        analysis_model = PO_NUMBER_TO_ANALYSIS_MODEL[len(self.physical_object_perimeters)]

        result = {}
        for heuristic_alias, physical_objects_heuristic in self.alias_to_standalone_heuristic.items():
            result[heuristic_alias] = analysis_model(
                manual_video=self.video,
                po_label_to_qualia_boolean_index={
                    heuristic.physical_object_label: heuristic.result for heuristic in physical_objects_heuristic
                },
            )

        return result

    @computed_field  # type: ignore[misc]
    @property
    def all_summary_series(self) -> list[pd.Series]:
        result = []
        for physical_objects_heuristics in self.alias_to_standalone_heuristic.values():
            for physical_objects_heuristic in physical_objects_heuristics:
                result.append(physical_objects_heuristic.summary_series)
        return result

    @computed_field  # type: ignore[misc]
    @cached_property
    def heuristic_to_object_alternation_sequence(self) -> dict[str, ConfinementSequence]:
        result = self.reader.confinement_sequence_defaultdict()
        for heuristic, heuristic_results in self.alias_to_standalone_heuristic.items():
            for perimeter_idx, po_qualia_heuristic in enumerate(heuristic_results, start=1):
                result[heuristic][po_qualia_heuristic.result] = perimeter_idx

        return dict(result)

    @computed_field  # type: ignore[misc]
    @cached_property
    def reduced_alternation_sequence(self) -> dict[str, ConfinementSequence]:
        result = {}
        for heuristic, object_alternation_sequence in self.heuristic_to_object_alternation_sequence.items():
            rrs = np.array(
                reduce_repeating_sequences(
                    object_alternation_sequence,
                    round(self.fps * self.minimum_seconds_tolerance),
                )
            )
            result[heuristic] = rrs[np.nonzero(rrs)]
        return result

    def generate_inspection_video(
        self,
        output_directory: Optional[DirectoryPath] = None,
        *,
        codec: Optional[str] = None,
        heuristics_to_use: Optional[list[str]] = None,
        **_kwargs,
    ) -> None:
        for heuristic_alias, heuristics in self.alias_to_combined_heuristic.items():
            if heuristics_to_use and heuristic_alias not in heuristics_to_use:
                continue

            make_inspection_video(
                video_frames=self.video.video_read_frames(),
                reader=self.reader,
                perimeter_to_boolean_index={po.perimeter: po.result for po in heuristics},
                output_file_path=self._video_file_name(output_directory, context_label=heuristic_alias),
                codec=codec,
            )

        for heuristic_alias, heuristics in self.alias_to_standalone_heuristic.items():
            if heuristics_to_use and heuristic_alias not in heuristics_to_use:
                continue

            perimeter_to_boolean_index = {}
            label_to_confinement_boolean_index = self.reader.confinement_index_defaultdict()
            label_to_quiver_rays = self.reader.coordinate_sequence_defaultdict()

            for heuristic in heuristics:
                perimeter_to_boolean_index.update(heuristic.perimeter_to_boolean_index)

                if issubclass(alias_to_heuristics_cls[heuristic_alias], ProximityMixin):
                    for label, proximity_boolean_index in heuristic.label_to_proximity_boolean.items():
                        label_to_confinement_boolean_index[label] = (
                            label_to_confinement_boolean_index[label] | proximity_boolean_index
                        )

                if issubclass(alias_to_heuristics_cls[heuristic_alias], RayMixin):
                    for label, ray_direction_points in heuristic.label_to_ray_vector_direction_points.items():
                        label_to_quiver_rays[label][heuristic.result] = ray_direction_points[heuristic.result]

            make_inspection_video(
                video_frames=self.video.video_read_frames(),
                reader=self.reader,
                perimeter_to_boolean_index=perimeter_to_boolean_index,
                label_to_confinement_boolean_index=label_to_confinement_boolean_index,
                label_to_quiver_rays=label_to_quiver_rays,
                output_file_path=self._video_file_name(output_directory, context_label=heuristic_alias),
                codec=codec,
            )

    @computed_field  # type: ignore[misc]
    @property
    def _analysis_series_list(self) -> list[pd.Series]:
        result = super()._analysis_series_list
        result.extend(
            (
                physical_object_analyser.analysis_series
                for physical_object_analyser in self.physical_object_analysers.values()
            )
        )
        result.extend(self.all_summary_series)
        return result
