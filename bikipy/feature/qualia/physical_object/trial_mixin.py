from abc import ABC, abstractmethod
from collections import defaultdict
from functools import cached_property

import pandas as pd
from pydantic import Field

from bikipy._constant import PHYSICAL_OBJECT_MAP_NAME
from bikipy.feature.qualia.physical_object.analysis.i import QualiaAnalysis
from bikipy.feature.qualia.physical_object.analysis.mapping import PO_NUMBER_TO_ANALYSIS_MODEL
from bikipy.feature.qualia.physical_object.heuristic.abc import QualiaHeuristic
from bikipy.feature.qualia.physical_object.heuristic.mapping import HEURISTIC_MAP
from bikipy.perimeter.base import SinglePerimeter
from bikipy.perimeter.mixin import PerimeterInstances, TrialWithPerimeterMixin
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
    def perimeters(self) -> list[SinglePerimeter, *PerimeterInstances]:
        # Inherit and append non-physical-object perimeters to the list
        return list(self.physical_object_perimeters)

    @cached_property
    def po_heuristic_alias_to_physical_objects_heuristic(
        self,
    ) -> dict[str, list[QualiaHeuristic]]:
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
        ) in self.po_heuristic_alias_to_physical_objects_heuristic.items():
            try:
                current_inspect_arg = self.inspect_arg / heuristic_alias
            except TypeError:
                # inspect_arg is a bool
                current_inspect_arg = self.inspect_arg

            po_label_to_qualia_boolean_index = {}
            for heuristic in physical_objects_heuristic:
                heuristic.plot()
                generic_inspection_finalization(
                    current_inspect_arg, f"{self.label}_{heuristic.label}_{heuristic_alias}.jpg"
                )
                po_label_to_qualia_boolean_index[heuristic.label] = heuristic.result

            result.append(
                analysis_model(
                    po_label_to_qualia_boolean_index=po_label_to_qualia_boolean_index, manual_video=self.video
                )
            )

        return result
