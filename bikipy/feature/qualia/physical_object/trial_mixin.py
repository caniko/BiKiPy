from abc import ABC, abstractmethod
from collections import defaultdict
from functools import cached_property

import numpy as np
import pandas as pd
from pydantic import Field

from bikipy._constant import INSPECT_FIG_FILE_FORMAT, PHYSICAL_OBJECT_MAP_NAME
from bikipy.behaviour.utils import reduce_repeating_sequences
from bikipy.feature.qualia.physical_object.analysis.i import QualiaAnalysis
from bikipy.feature.qualia.physical_object.analysis.mapping import (
    PO_NUMBER_TO_ANALYSIS_MODEL,
)
from bikipy.feature.qualia.physical_object.heuristic.abc import QualiaHeuristic
from bikipy.feature.qualia.physical_object.heuristic.mapping import HEURISTIC_MAP
from bikipy.perimeter.base import SinglePerimeter
from bikipy.perimeter.helper.confinement import ConfinementSequence
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
    def po_heuristic_to_heuristic_physical_objects(self) -> dict[str, list[QualiaHeuristic]]:
        """
        Note that the values being lists are bijective counterparts to
        self.physical_object_perimeters sequence of perimeters.

        :return:
        """
        result = defaultdict(list)
        for heuristic_alias, heuristic_config in self.project_kit_config[PHYSICAL_OBJECT_MAP_NAME].items():
            for perimeter in self.physical_object_perimeters:
                result[heuristic_alias].append(
                    HEURISTIC_MAP[heuristic_alias](
                        perimeter=perimeter, reader=self.reader, manual_video=self.video, **heuristic_config
                    )
                )
        return dict(result)

    @cached_property
    def physical_object_analysers(self) -> list[QualiaAnalysis]:
        analysis_model = PO_NUMBER_TO_ANALYSIS_MODEL[len(self.physical_object_perimeters)]

        result = []
        for (
            heuristic_alias,
            physical_objects_heuristic,
        ) in self.po_heuristic_to_heuristic_physical_objects.items():
            try:
                current_inspect_arg = self.inspect_arg / heuristic_alias
            except TypeError:
                # inspect_arg is a bool
                current_inspect_arg = self.inspect_arg

            po_to_qualia_boolean_index = {}
            for heuristic in physical_objects_heuristic:
                po_to_qualia_boolean_index[heuristic.po_label] = heuristic.result
                if self.inspect_arg:
                    heuristic.plot()
                    generic_inspection_finalization(
                        current_inspect_arg,
                        f"{self.label}_{heuristic.po_label}_{heuristic_alias}{INSPECT_FIG_FILE_FORMAT}",
                    )

            result.append(
                analysis_model(po_label_to_qualia_boolean_index=po_to_qualia_boolean_index, manual_video=self.video)
            )

        return result

    @property
    def all_summary_series(self) -> list[pd.Series]:
        result = []
        for physical_objects_heuristics in self.po_heuristic_to_heuristic_physical_objects.values():
            for physical_objects_heuristic in physical_objects_heuristics:
                result.append(physical_objects_heuristic.summary_series)
        return result

    @cached_property
    def po_heuristic_to_object_alternation_sequence(self) -> dict[str, ConfinementSequence]:
        result = self.reader.confinement_sequence_defaultdict()
        for heuristic, po_heuristic_results in self.po_heuristic_to_heuristic_physical_objects.items():
            for perimeter_idx, po_qualia_heuristic in enumerate(po_heuristic_results, start=1):
                result[heuristic][po_qualia_heuristic.result] = perimeter_idx

        return dict(result)

    @cached_property
    def reduced_alternation_sequence(self) -> dict[str, ConfinementSequence]:
        result = {}
        for heuristic, object_alternation_sequence in self.po_heuristic_to_object_alternation_sequence.items():
            rrs = reduce_repeating_sequences(
                object_alternation_sequence, round(self.video.fps * self.minimum_seconds_tolerance)
            )
            result[heuristic] = rrs[np.nonzero(rrs)]
        return result
