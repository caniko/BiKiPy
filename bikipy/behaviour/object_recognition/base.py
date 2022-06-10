from abc import ABC
from functools import cached_property
from logging import getLogger
from typing import ClassVar, Hashable, Optional

import pandas as pd
from pydantic import Field, root_validator

from bikipy.behaviour.mixin.physical_object import (
    PhysicalObjectExperimentMixin,
    PhysicalObjectTrialMixin,
)
from bikipy.behaviour.rectangle.square import (
    SquareEnclosedExperiment,
    SquareEnclosedTrial,
)
from bikipy.feature.physical_object.field import ObjectField

logger = getLogger(__name__)


class ObjectRecognitionExperiment(SquareEnclosedExperiment, PhysicalObjectExperimentMixin):
    global_object_field: Optional[ObjectField] = Field(
        description="""
        ObjectField to be used for all trials. Most common use case is when the ObjectField has its 
        object-presence sequences are defined by dictionary.
        """
    )
    first_stage_has_no_object: ClassVar[bool] = False

    def trial_keyword_arguments(self, trial_id: Hashable) -> dict:
        upstream_kwargs = super().trial_keyword_arguments(trial_id)

        if self.first_stage_has_no_object and upstream_kwargs["stage"] == 0:
            return upstream_kwargs

        result = {
            **upstream_kwargs,
            **self._physical_object_keyword_arguments,
            "perimeter_border_normal_metric_magnitude": self.perimeter_border_normal_metric_magnitude,
        }

        return result

    @root_validator
    def global_object_field_and_id_to_object_field_mutually_exclusive(cls, values):
        if values["global_object_field"] is not None and values["id_to_object_field"] is not None:
            msg = "global_object_field and id_to_object_field are mutually exclusive"
            raise ValueError(msg)
        return values


class ObjectRecognitionHabituationTrial(SquareEnclosedTrial):
    """The purpose of this stage is to generate reference data for proceeding experiments with objects."""

    trial_stage_index: ClassVar[Optional[int]] = 0
    trial_label: ClassVar[str] = "Habituation"


class GenericObjectRecognitionTrial(SquareEnclosedTrial, PhysicalObjectTrialMixin, ABC):
    general_feature_headers: ClassVar[list] = Field(default_factory=list)
    object_feature_headers: ClassVar[list] = Field(default_factory=list)

    """
    Remember to implement both feature headers and summary for each generic feature you plan to add.
    Otherwise, the pandas interaction will raise an error or lead to unexpected results.
    """

    @cached_property
    def feature_headers(self) -> list:
        general_feature_headers = []
        # Add generic feature headers ======================================
        if self.number_of_physical_objects == 2:
            general_feature_headers.append("Absolute pair discrimination")
        # ==================================================================
        result = pd.MultiIndex.from_product([["General"], general_feature_headers + self.general_feature_headers])
        return result + pd.MultiIndex.from_product(
            [
                list(self.physical_object_labels),
                [
                    # Add object feature headers =========================================================
                    "Seconds observing",
                    # ====================================================================================
                ]
                + self.object_feature_headers,
            ]
        )

    @property
    def general_features(self) -> list:
        generic = []
        # Add generic feature headers ================================================================
        if self.number_of_physical_objects == 2:
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
        assert len(self.object_features) % self.number_of_physical_objects == 0
        return self.general_features + self.object_features
