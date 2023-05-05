from abc import ABC, abstractmethod
from collections import defaultdict
from functools import cached_property
from typing import Optional, TypeVarTuple

import pandas as pd
from projectkit.model.project import ProjectKitDownstreamBranchingMixin
from pydantic import validate_arguments, Field
from schemantic.model.schema import GroupSchema

from bikipy.feature.qualia.physical_object.analysis.i import QualiaAnalysis
from bikipy.feature.qualia.physical_object.analysis.mapping import PO_NUMBER_TO_ANALYSIS_MODEL
from bikipy.feature.qualia.physical_object.qualia_heuristic.abc import QualiaProfile
from bikipy.feature.qualia.physical_object.qualia_heuristic.mapping import PROFILE_MAP
from bikipy.perimeter.base import SinglePerimeter
from bikipy.perimeter.mixin import PerimeterInstances, TrialWithPerimeterMixin
from bikipy.utils.plot.inspect import generic_inspection_finalization

PhysicalObjectInstances = TypeVarTuple("PhysicalObjectInstances")


class PhysicalObjectTrialMixin(TrialWithPerimeterMixin, ProjectKitDownstreamBranchingMixin, ABC):
    qualia_profiles_combination_equations: list[str, ...] = Field(
        # TODO: Implement
        default_factory=list,
        description="Performs analysis by combining qualia heuristic result with respect to "
        "the defined logical method AND/OR using & or | respectively.",
    )

    schemantic_branch_config_map_key = "qualia_profiler"

    @property
    @abstractmethod
    def physical_object_perimeters(self) -> tuple[SinglePerimeter, *PhysicalObjectInstances]:
        ...

    @classmethod
    @validate_arguments
    def project_kit_branch_schema(
        cls, qualia_definition_profile: Optional[list[str, ...]] = None, **kwargs
    ) -> list[GroupSchema, ...]:
        """

        :param qualia_definition_profile:
        :param kwargs:
        :return:
        """
        upstream = super().project_kit_branch_schema(**kwargs)
        if not qualia_definition_profile:
            return upstream

        models = set()
        assert qualia_definition_profile
        for heuristic in qualia_definition_profile:
            try:
                models.add(PROFILE_MAP[heuristic])
            except KeyError:
                msg = (
                    f"The defined heuristic key, {heuristic}, is not defined. "
                    f"Choose from the following: {', '.join(iter(PROFILE_MAP))}"
                )
                raise KeyError(msg)

        upstream.append(GroupSchema.from_models(models=models, mapping_name=cls.schemantic_branch_config_map_key))
        return upstream

    @cached_property
    def perimeters(self) -> list[SinglePerimeter, *PerimeterInstances]:
        # Inherit and append non-physical-object perimeters to the list
        return list(self.physical_object_perimeters)

    @cached_property
    def po_profile_alias_to_profiled_physical_objects(self) -> dict[str, list[QualiaProfile, *PhysicalObjectInstances]]:
        result = defaultdict(list)
        for profile_alias, profile_config in self.project_kit_get_branch_config.items():
            for perimeter in self.physical_object_perimeters:
                result[profile_alias].append(
                    PROFILE_MAP[profile_alias](perimeter=perimeter, reader=self.reader, **profile_config)
                )
        return dict(result)

    @cached_property
    def physical_object_analysers(self) -> list[QualiaAnalysis, ...]:
        analysis_model = PO_NUMBER_TO_ANALYSIS_MODEL[len(self.physical_object_perimeters)]

        result = []
        for profile_alias, profiled_physical_objects in self.po_profile_alias_to_profiled_physical_objects.items():
            try:
                current_inspect_arg = self.inspect_arg / profile_alias
            except TypeError:
                # inspect_arg is a bool
                current_inspect_arg = self.inspect_arg

            po_label_to_qualia_boolean_index = {}
            for profiled in profiled_physical_objects:
                profiled.plot()
                generic_inspection_finalization(
                    current_inspect_arg, f"{self.int_id}_{profiled.label}_{profile_alias}.jpg"
                )
                po_label_to_qualia_boolean_index[profiled.label] = profiled.result

            result.append(analysis_model(po_label_to_qualia_boolean_index=po_label_to_qualia_boolean_index))

        return result

    @property
    def _analysis_series_list(self) -> list[pd.Series, ...]:
        upstream = super()._analysis_series_list
        upstream.extend(
            (physical_object_analyser.analysis_series for physical_object_analyser in self.physical_object_analysers)
        )
        return upstream
