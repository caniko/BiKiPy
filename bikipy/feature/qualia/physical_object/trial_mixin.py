from abc import ABC, abstractmethod
from collections import defaultdict
from functools import cached_property
from typing import Optional

import numpy as np
import pandas as pd
from pydantic import DirectoryPath, Field

from bikipy._constant import INSPECT_FIG_FILE_FORMAT, PHYSICAL_OBJECT_MAP_NAME
from bikipy.analysis.video import make_inspection_video
from bikipy.behaviour.utils import reduce_repeating_sequences
from bikipy.core.typing import ConfinementSequence
from bikipy.feature.qualia.physical_object.analysis.i import QualiaAnalysis
from bikipy.feature.qualia.physical_object.analysis.mapping import (
    PO_NUMBER_TO_ANALYSIS_MODEL,
)
from bikipy.feature.qualia.physical_object.heuristic.abc import (
    ProximityMixin,
    QualiaHeuristic,
    RayMixin,
)
from bikipy.feature.qualia.physical_object.heuristic.mapping import ALIAS_TO_HEURISTIC_CLS
from bikipy.feature.qualia.physical_object.merge_parser import parse_heuristic_merge_equation
from bikipy.perimeter.base import SinglePerimeter
from bikipy.perimeter.mixin import TrialWithPerimeterMixin
from bikipy.utils.plot.inspect import generic_inspection_finalization


class PhysicalObjectTrialMixin(TrialWithPerimeterMixin, ABC):
    qualia_heuristics_combination_equations: list[str] = Field(
        # TODO: Implement
        default_factory=list,
        description="Performs analysis by combining qualia heuristic result with respect to "
        "the defined logical method AND/OR using & or | respectively.",
    )

    @property
    @abstractmethod
    def physical_object_perimeters(self) -> tuple[SinglePerimeter, ...]:
        ...

    @cached_property
    def perimeters(self):
        # Inherit and append non-physical-object perimeters to the list
        return tuple(self.physical_object_perimeters)

    @cached_property
    def alias_to_heuristic_physical_objects(self) -> dict[str, list[QualiaHeuristic]]:
        """
        Note that the values being lists are bijective counterparts to
        self.physical_object_perimeters sequence of perimeters.

        :return:
        """
        result = defaultdict(list)

        for heuristic_alias, heuristic_config in self.project_kit_config[PHYSICAL_OBJECT_MAP_NAME].items():
            heuristic_inspect_arg = (
                self.inspect_arg if isinstance(self.inspect_arg, bool) else self.inspect_arg / heuristic_alias
            )
            for perimeter in self.physical_object_perimeters:
                heuristic = ALIAS_TO_HEURISTIC_CLS[heuristic_alias](
                    perimeter=perimeter, reader=self.reader, manual_video=self.video, **heuristic_config
                )
                if self.inspect_arg:
                    heuristic.plot()
                    generic_inspection_finalization(
                        heuristic_inspect_arg,
                        f"{self.label}_{heuristic.po_label}_{heuristic_alias}{INSPECT_FIG_FILE_FORMAT}",
                    )
                result[heuristic_alias].append(heuristic)

        for heuristic_alias, heuristic_equation in self.alias_to_qualia_heuristics_combination_equations:
            result[heuristic_alias] = parse_heuristic_merge_equation(heuristic_equation, result)

        return dict(result)

    @cached_property
    def physical_object_analysers(self) -> dict[str, QualiaAnalysis]:
        analysis_model = PO_NUMBER_TO_ANALYSIS_MODEL[len(self.physical_object_perimeters)]

        result = {}
        for heuristic_alias, physical_objects_heuristic in self.alias_to_heuristic_physical_objects.items():
            result[heuristic_alias] = analysis_model(
                manual_video=self.video,
                po_label_to_qualia_boolean_index={
                    heuristic.po_label: heuristic.result for heuristic in physical_objects_heuristic
                },
            )

        return result

    @property
    def alias_to_merged_heuristics(self) -> dict[str, QualiaAnalysis]:
        pass

    @property
    def all_summary_series(self) -> list[pd.Series]:
        result = []
        for physical_objects_heuristics in self.alias_to_heuristic_physical_objects.values():
            for physical_objects_heuristic in physical_objects_heuristics:
                result.append(physical_objects_heuristic.summary_series)
        return result

    @cached_property
    def heuristic_to_object_alternation_sequence(self) -> dict[str, ConfinementSequence]:
        result = self.reader.confinement_sequence_defaultdict()
        for heuristic, heuristic_results in self.alias_to_heuristic_physical_objects.items():
            for perimeter_idx, po_qualia_heuristic in enumerate(heuristic_results, start=1):
                result[heuristic][po_qualia_heuristic.result] = perimeter_idx

        return dict(result)

    @cached_property
    def reduced_alternation_sequence(self) -> dict[str, ConfinementSequence]:
        result = {}
        for heuristic, object_alternation_sequence in self.heuristic_to_object_alternation_sequence.items():
            rrs = reduce_repeating_sequences(
                object_alternation_sequence, round(self.video.fps * self.minimum_seconds_tolerance)
            )
            result[heuristic] = rrs[np.nonzero(rrs)]
        return result

    def generate_inspection_video(
        self, output_directory: Optional[DirectoryPath] = None, codec: Optional[str] = None
    ) -> None:
        for heuristic_alias, physical_objects in self.alias_to_heuristic_physical_objects.items():
            perimeter_to_boolean_index = self.reader.confinement_index_defaultdict
            label_to_boolean_index = self.reader.confinement_index_defaultdict
            label_to_quiver_rays = self.reader.coordinate_sequence_defaultdict

            for physical_object in physical_objects:
                perimeter_to_boolean_index[physical_object.perimeter] = (
                    perimeter_to_boolean_index[physical_object.perimeter] | physical_object.result
                )
                if issubclass(ALIAS_TO_HEURISTIC_CLS[heuristic_alias], ProximityMixin):
                    for label, proximity_boolean_index in physical_object.label_to_proximity_boolean.items():
                        label_to_boolean_index[label] = label_to_boolean_index[label] | proximity_boolean_index
                if issubclass(ALIAS_TO_HEURISTIC_CLS[heuristic_alias], RayMixin):
                    for label, ray_direction_points in physical_object.label_to_ray_vector_direction_points.items():
                        label_to_quiver_rays[label][physical_object.result] = ray_direction_points[
                            physical_object.result
                        ]

            make_inspection_video(
                video_frames=self.video.video_read_frames(),
                reader=self.reader,
                perimeter_to_boolean_index={po.perimeter: po.result for po in physical_objects},
                label_to_boolean_index=label_to_boolean_index,
                label_to_quiver_rays=label_to_quiver_rays,
                output_file_path=self._video_file_name(output_directory),
                codec=codec,
            )

    @property
    def _analysis_series_list(self) -> list[pd.Series]:
        result = super()._analysis_series_list
        result.extend(
            (physical_object_analyser.analysis_series for physical_object_analyser in self.physical_object_analysers)
        )
        result.extend(self.all_summary_series)
        return result
