from abc import ABC, abstractmethod
from functools import cached_property
from typing import Any, ClassVar, Literal, Optional

import pandas as pd
from pydantic import Field
from schemantic.model.project import SchemanticBranchingMixin

from bikipy.feature.physical_object.analysis import (
    PhysicalObjectSetCLS,
    PhysicalObjectSet,
    GenericPhysicalObjectSet,
    PhysicalObjectSetAnalysis,
)
from bikipy.feature.physical_object.component.abc import ObservationComponent
from bikipy.perimeter.base import SinglePerimeter
from bikipy.perimeter.mixin import TrialWithPerimeterMixin


class PhysicalObjectTrialMixin(TrialWithPerimeterMixin, SchemanticBranchingMixin, ABC):
    qualia_definition_profile: Literal["rodent", None]
    qualia_definition_sequence: Optional[list[ObservationComponent, ...]] = Field(default_factory=list)

    @property
    @abstractmethod
    def physical_object_perimeters(self) -> tuple[SinglePerimeter, ...]:
        ...

    def __getitem__(self, item):
        return self.label_to_physical_object[item]

    @classmethod
    def _schemantic_branch_schema(cls, **kwargs) -> "GroupSchema":
        """

        :param kwargs:
        :return:
        """
        ...

    def physical_object_observation_qualia(self):
        if self.qualia_definition_profile:
            try:
                return

    @cached_property
    def perimeters(self) -> list[SinglePerimeter, ...]:
        # Inherit and append non-physical-object perimeters to the list
        return list(self.physical_object_perimeters)

    @property
    def _trial_analysis_series_list(self) -> list[pd.Series]:
        upstream_list = super()._trial_analysis_series_list
        upstream_list.append(self.physical_object_set.feature_summary)
        return upstream_list

    @cached_property
    def label_to_physical_object(self) -> dict:
        return {physical_object.label: physical_object for physical_object in self.physical_objects}

    def gaze_analysis(self) -> PhysicalObjectSetAnalysis:
        return PhysicalObjectSetAnalysis(
            analysis_label="Gaze",
            physical_object_label_to_observation_boolean_index={
                physical_object.label: physical_object.attention_observance_boolean_index
                for physical_object in self.physical_objects
            },
            manual_video=self.video,
        )

    def proximity_analysis(self) -> PhysicalObjectSetAnalysis:
        return PhysicalObjectSetAnalysis(
            analysis_label="Proximity",
            physical_object_label_to_observation_boolean_index={
                physical_object.label: physical_object.attention_proximity_boolean_index
                for physical_object in self.physical_objects
            },
            manual_video=self.video,
        )

    @property
    def feature_summary(self) -> pd.Series:
        return pd.concat(
            (
                # *(analysis_object.feature_summary for analysis_object in self.analysis_objects),
                *(physical_object.summary for physical_object in self.physical_objects),
            )
        )

    @property
    def seconds_observing(self) -> float:
        return self.analysis_objects[0].total_seconds_observing

    def plot(self, ax: Any = None):
        if not ax:
            fix, ax = self.video.subplots()
        for physical_objects in self.physical_objects:
            ax = physical_objects.perimeter.plot(ax=ax)

        return ax
