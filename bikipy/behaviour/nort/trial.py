from functools import cached_property
from logging import getLogger
from typing import ClassVar, Optional

from pydantic import BaseModel, Field

from bikipy.behaviour.mixins.misc import OpenFieldTrialMixin
from bikipy.behaviour.mixins.physical_object import PhysicalObjectTrialMixin
from bikipy.behaviour.nort.experiment import NortField
from bikipy.behaviour.square import SquareEnclosedTrial

logger = getLogger(__name__)


class NortHabituationTrial(SquareEnclosedTrial, OpenFieldTrialMixin):
    """
    NORT experiment without any objects. The purpose of this test is to generate
    reference data for future NORT experiments.
    """

    trial_sequence_index: ClassVar[Optional[int]] = 0
    trial_label: ClassVar[str] = "habituation"

    trial_has_feature_frame: ClassVar[bool] = False


class NortOpenField(NortHabituationTrial):
    pass


class NortFieldMixin(BaseModel):
    nort_field: NortField


class NortTrainingTrial(SquareEnclosedTrial, PhysicalObjectTrialMixin, NortFieldMixin):
    trial_sequence_index: ClassVar[Optional[int]] = 1
    trial_label: ClassVar[str] = "training"

    trial_has_feature_frame: ClassVar[bool] = True

    @cached_property
    def physical_object_set(self):
        return self.nort_field.training_set(self._physical_object_keyword_arguments)

    @property
    def physical_object_variable(self):
        return self.physical_object_set.physical_objects[0]

    @property
    def physical_object_constant(self):
        return self.physical_object_set.physical_objects[1]

    @property
    def feature_summary_column(self) -> list:
        return ["Seconds observing"]

    @property
    def feature_summary_row(self):
        return [self.physical_object_set.seconds_observing]


class NortNoveltyTrial(SquareEnclosedTrial, PhysicalObjectTrialMixin, NortFieldMixin):
    trial_sequence_index: ClassVar[Optional[int]] = 2
    trial_label: ClassVar[str] = "novelty"

    trial_has_feature_frame: ClassVar[bool] = True

    @cached_property
    def physical_object_set(self):
        return self.nort_field.novelty_set(self._physical_object_keyword_arguments)

    @property
    def physical_object_novel(self):
        return self.physical_object_set.physical_objects[0]

    @property
    def physical_object_constant(self):
        return self.physical_object_set.physical_objects[1]

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
    def feature_summary_column(self) -> list:
        return [
            "Absolute discrimination",
            "Discrimination index",
            "Novelty index",
        ]

    @property
    def feature_summary_row(self):
        return [
            self.physical_object_set.absolute_discrimination,
            self.discrimination_index,
            self.novelty_preference,
        ]


CLASS_NAME_VS_CLASS = {
    "habituation": NortHabituationTrial,
    "training": NortTrainingTrial,
    "novelty": NortNoveltyTrial,
}
