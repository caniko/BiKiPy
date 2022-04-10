from functools import cached_property
from logging import getLogger
from typing import Optional, ClassVar

import pandas as pd
from pydantic import Field

from bikipy.behaviour.mixin.physical_object import PhysicalObjectExperimentMixin, PhysicalObjectBaseMixin
from bikipy.behaviour.rectangle.square import (
    SquareEnclosedExperiment,
    SquareEnclosedTrial,
)
from bikipy.feature.physical_object.core import PhysicalObjectSet
from bikipy.feature.physical_object.field import ObjectField

logger = getLogger(__name__)


class ObjectRecognitionExperiment(
    SquareEnclosedExperiment, PhysicalObjectExperimentMixin
):
    id_vs_object_field: Optional[dict[int, ObjectField]] = Field(
        None, description="Label of the nose in the df"
    )
    first_stage_has_no_object: ClassVar[bool] = True

    def trial_keyword_arguments(self, trial_id: int) -> dict:
        upstream_kwargs = super().trial_keyword_arguments(trial_id)

        if self.first_stage_has_no_object and upstream_kwargs["stage"] == 0:
            return upstream_kwargs

        return {
            **upstream_kwargs,
            **self._physical_object_keyword_arguments,
            "perimeter_border_normal_metric_magnitude": self.perimeter_border_normal_metric_magnitude,
            "object_field": self.id_vs_object_field[upstream_kwargs["field_id"]],
        }


class ObjectRecognitionHabituationTrial(SquareEnclosedTrial):
    """The purpose of this stage is to generate reference data for proceeding experiments with objects."""
    trial_sequence_index: ClassVar[Optional[int]] = 0
    trial_label: ClassVar[str] = "Habituation"


class GenericObjectRecognitionTrial(SquareEnclosedTrial, PhysicalObjectBaseMixin):
    object_field: ObjectField

    general_feature_headers: ClassVar[list] = Field(default_factory=list)
    object_feature_headers: ClassVar[list] = Field(default_factory=list)

    @cached_property
    def physical_object_set(self) -> PhysicalObjectSet:
        return self.object_field.derive_physical_object_set(
            self.trial_sequence_index,
            **self._physical_object_keyword_arguments
        )

    @cached_property
    def number_of_objects(self) -> int:
        return len(self.object_field)

    """
    Remember to implement both feature headers and summary for each generic feature you plan to add.
    Otherwise, the pandas interaction will raise an error or lead to unexpected results.
    """

    @cached_property
    def feature_headers(self) -> list:
        general_feature_headers = []
        # Add generic feature headers ======================================
        if self.number_of_objects == 2:
            general_feature_headers.append("Absolute pair discrimination")
        # ==================================================================
        result = pd.MultiIndex.from_product([
            ["General"],
            general_feature_headers + self.general_feature_headers
        ])
        return result + pd.MultiIndex.from_product([
            list(self.object_field.labels),
            [
                # Add object feature headers =========================================================
                "Seconds observing",
                # ====================================================================================
            ] + self.object_feature_headers
        ])

    @property
    def general_features(self) -> list:
        generic = []
        # Add generic feature headers ================================================================
        if self.number_of_objects == 2:
            generic.append(self.physical_object_set)
        # ============================================================================================
        return generic

    @property
    def object_features(self) -> list:
        return [
            # Add object feature headers =============================================================
            *self.physical_object_set.object_specific_observation.values()
            # ========================================================================================
        ]

    @property
    def feature_summary_row(self):
        # Don't inherit this! Inherit general_features and object_features individually
        assert len(self.object_features) % self.number_of_objects == 0
        return self.general_features + self.object_features
