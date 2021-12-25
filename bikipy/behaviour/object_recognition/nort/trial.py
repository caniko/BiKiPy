from abc import ABC
from functools import cached_property
from logging import getLogger
from typing import ClassVar, Optional

import pandas as pd
from pydantic import BaseModel

from bikipy.behaviour.mixin.base import FeaturefullTrialMixin
from bikipy.behaviour.mixin.misc import OpenFieldTrialMixin
from bikipy.behaviour.mixin.physical_object import PhysicalObjectTrialMixin
from bikipy.behaviour.object_recognition.nort.experiment import NortField
from bikipy.behaviour.square import SquareEnclosedTrial

logger = getLogger(__name__)


class NortHabituationTrial(SquareEnclosedTrial, OpenFieldTrialMixin):
    """
    NORT experiment without any objects. The purpose of this test is to generate
    reference data for future NORT experiments.
    """

    trial_sequence_index: ClassVar[Optional[int]] = 0
    trial_label: ClassVar[str] = "Habituation"

    trial_has_feature_frame: ClassVar[bool] = False


class NortOpenField(NortHabituationTrial):
    pass


class NortPhysicalObjectFieldMixin(PhysicalObjectTrialMixin, ABC):
    nort_field: NortField

    @property
    def physical_object_constant(self):
        return self.physical_object_set.physical_objects[1]


class NortTrainingTrial(SquareEnclosedTrial, NortPhysicalObjectFieldMixin, FeaturefullTrialMixin):
    trial_sequence_index: ClassVar[Optional[int]] = 1
    trial_label: ClassVar[str] = "Training"

    feature_summary_column: ClassVar[list] = [("Training", "Seconds observing")]

    @cached_property
    def physical_object_set(self):
        return self.nort_field.training_set(self._physical_object_keyword_arguments)

    @property
    def physical_object_variable(self):
        return self.physical_object_set.physical_objects[0]

    @property
    def feature_summary_row(self):
        return [self.physical_object_set.seconds_observing]


class NortNoveltyTrial(SquareEnclosedTrial, NortPhysicalObjectFieldMixin, FeaturefullTrialMixin):
    trial_sequence_index: ClassVar[Optional[int]] = 2
    trial_label: ClassVar[str] = "Novelty"

    feature_summary_column: ClassVar[list] = list(
        pd.MultiIndex.from_product(
            [
                ["Novelty"],
                [
                    "Absolute discrimination",
                    "Discrimination index",
                    "Novelty preference",
                    "Object bias score",
                ],
            ]
        )
    )

    @cached_property
    def physical_object_set(self):
        return self.nort_field.novelty_set(self._physical_object_keyword_arguments)

    @property
    def physical_object_novel(self):
        return self.physical_object_set.physical_objects[0]

    @cached_property
    def discrimination_index(self):
        return (
            self.physical_object_set.absolute_discrimination / self.experiment_seconds
        )

    @cached_property
    def novelty_preference(self):
        return (
            100.0
            * self.physical_object_novel.attention_filtered_seconds_observing
            / self.experiment_seconds
        )

    @property
    def feature_summary_row(self):
        return [
            self.physical_object_set.absolute_discrimination,
            self.discrimination_index,
            self.novelty_preference,
            self.physical_object_set.object_bias_score[1],
        ]


CLASS_NAME_VS_CLASS = {
    "habituation": NortHabituationTrial,
    "training": NortTrainingTrial,
    "novelty": NortNoveltyTrial,
}
