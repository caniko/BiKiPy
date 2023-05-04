from abc import ABC, abstractmethod
from functools import cached_property
from typing import Literal, Optional

import pandas as pd
from pydantic import Field, validate_arguments
from schemantic.model.project import SchemanticBranchingMixin
from schemantic.model.schema import GroupSchema

from bikipy.feature.physical_object.qualia.mapping import PROFILE_MAP
from bikipy.feature.physical_object.qualia.profile.abc import QualiaProfile
from bikipy.perimeter.base import SinglePerimeter
from bikipy.perimeter.mixin import TrialWithPerimeterMixin


class PhysicalObjectTrialMixin(TrialWithPerimeterMixin, SchemanticBranchingMixin, ABC):
    qualia_definition_sequence: Optional[list[QualiaProfile, ...]] = Field(default_factory=list)
    qualia_definition_profile: Literal["rodent", None]

    @property
    @abstractmethod
    def physical_object_perimeters(self) -> tuple[SinglePerimeter, ...]:
        ...

    @classmethod
    @validate_arguments
    def schemantic_branch_schema(cls, *, qualia_definition_profile: list[str, ...], **kwargs) -> list[GroupSchema, ...]:
        """

        :param qualia_definition_profile:
        :param kwargs:
        :return:
        """
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

        upstream = super().schemantic_branch_schema(**kwargs)
        upstream.append(GroupSchema.from_models(models=models))
        return upstream

    @cached_property
    def perimeters(self) -> list[SinglePerimeter, ...]:
        # Inherit and append non-physical-object perimeters to the list
        return list(self.physical_object_perimeters)

    @cached_property
    def label_to_physical_object(self) -> dict:
        return {physical_object.label: physical_object for physical_object in self.physical_objects}

    @property
    def feature_summary(self) -> pd.Series:
        return pd.concat(
            (
                # *(analysis_object.feature_summary for analysis_object in self.analysis_objects),
                *(physical_object.summary for physical_object in self.physical_objects),
            )
        )
