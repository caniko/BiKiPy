from abc import ABC, abstractmethod
from collections import defaultdict
from functools import cached_property
from typing import Literal, Optional, TypeVarTuple

import pandas as pd
from projectkit.model.project import ProjectKitDownstreamBranchingMixin
from pydantic import Field, validate_arguments
from schemantic.model.schema import GroupSchema

from bikipy.feature.physical_object.analysis.i import QualiaAnalysis
from bikipy.feature.physical_object.analysis.mapping import PO_NUMBER_TO_ANALYSIS_MODEL
from bikipy.feature.physical_object.qualia_profiler.mapping import PROFILE_MAP
from bikipy.feature.physical_object.qualia_profiler.abc import QualiaProfile
from bikipy.perimeter.base import SinglePerimeter
from bikipy.perimeter.mixin import TrialWithPerimeterMixin, PerimeterInstances


PhysicalObjectInstances = TypeVarTuple("PhysicalObjectInstances")


class PhysicalObjectTrialMixin(TrialWithPerimeterMixin, ProjectKitDownstreamBranchingMixin, ABC):
    schemantic_branch_config_map_key = "qualia_profiler"

    @property
    @abstractmethod
    def physical_object_perimeters(self) -> tuple[SinglePerimeter, *PhysicalObjectInstances]:
        ...

    @classmethod
    @validate_arguments
    def project_kit_branch_schema(cls, qualia_definition_profile: Optional[list[str, ...]] = None, **kwargs) -> list[GroupSchema, ...]:
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
        for profile in qualia_definition_profile:
            try:
                models.add(PROFILE_MAP[profile])
            except KeyError:
                msg = (
                    f"The defined profile key, {profile}, is not defined. "
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
                result[profile_alias].append(PROFILE_MAP[profile_alias](perimeter=perimeter, reader=self.reader, **profile_config))
        return dict(result)

    @cached_property
    def physical_object_analyser(self) -> list[QualiaAnalysis, ...]:
        analysis_model = PO_NUMBER_TO_ANALYSIS_MODEL[len(self.physical_object_perimeters)]

        result = []
        for profile_alias, profiled_physical_objects in self.po_profile_alias_to_profiled_physical_objects.items():
            result.append(analysis_model(
                physical_object_label_to_observation_boolean_index={
                    profiled.perimeter.label: profiled.result
                }
            ))

        return result

    @property
    def _analysis_series_list(self) -> list[pd.Series, ...]:
        return pd.concat(
            (
                # *(analysis_object.feature_summary for analysis_object in self.analysis_objects),
                *(physical_object.summary for physical_object in self.physical_objects),
            )
        )
