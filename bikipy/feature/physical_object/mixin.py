from abc import ABC, abstractmethod
from functools import cached_property
from typing import TYPE_CHECKING, Any, Literal, Optional

import pandas as pd
from pydantic import Field, validate_arguments
from schemantic.model.project import SchemanticBranchingMixin

from bikipy.feature.physical_object.qualia.component.abc import QualiaComponent
from bikipy.perimeter.base import SinglePerimeter
from bikipy.perimeter.mixin import TrialWithPerimeterMixin

if TYPE_CHECKING:
    from schemantic.model.schema import GroupSchema


class PhysicalObjectTrialMixin(TrialWithPerimeterMixin, SchemanticBranchingMixin, ABC):
    qualia_definition_sequence: Optional[list[QualiaComponent, ...]] = Field(default_factory=list)
    qualia_definition_profile: Literal["rodent", None]

    @property
    @abstractmethod
    def physical_object_perimeters(self) -> tuple[SinglePerimeter, ...]:
        ...

    def __getitem__(self, item):
        return self.label_to_physical_object[item]

    @classmethod
    @validate_arguments
    def _schemantic_branch_schema(cls, qualia_definition_sequence: Optional[list[QualiaComponent, ...]] = None,
    qualia_definition_profile: Literal["rodent", None] = None) -> "GroupSchema":
        """

        :param kwargs:
        :return:
        """
        if qualia_definition_profile:


    def physical_object_observation_qualia(self):
        if self.qualia_definition_profile:
            try:
                return

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
