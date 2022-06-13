from functools import cached_property
from logging import getLogger
from typing import ClassVar, Optional

from bikipy.behaviour.mixin.physical_object import (
    PhysicalObjectHabituationTrialMixin,
    RectanglePhysicalObjectExperiment,
    RectanglePhysicalObjectTrial,
)
from bikipy.behaviour.rectangle import RectangleEnclosedTrial
from bikipy.perimeter.base import AnyPerimeter

logger = getLogger(__name__)


class NortHabituationTrial(RectangleEnclosedTrial, PhysicalObjectHabituationTrialMixin):
    pass


class NortOpenField(NortHabituationTrial):
    pass


class NortTrainingTrial(RectanglePhysicalObjectTrial):
    variable: AnyPerimeter
    familiar: AnyPerimeter

    physical_object_labels: ClassVar[list[str, ...]] = ["variable", "familiar"]

    experiment_sequence_index: ClassVar[Optional[int]] = 1
    trial_label: ClassVar[str] = "Training"

    @property
    def all_physical_object_perimeters(self):
        return self.variable, self.familiar


class NortNoveltyTrial(RectanglePhysicalObjectTrial):
    novel: AnyPerimeter
    familiar: AnyPerimeter

    physical_object_labels: ClassVar[list[str, ...]] = ["novel", "familiar"]

    experiment_sequence_index: ClassVar[Optional[int]] = 2
    trial_label: ClassVar[str] = "Novelty"

    @classmethod
    @property
    def feature_headers(cls) -> list[str]:
        return super().feature_headers + [
            "Absolute discrimination",
            "Discrimination index",
            "Novelty preference",
            "Object bias score",
        ]

    @property
    def feature_summary_row(self):
        return super().feature_summary_row + [
            self.nort_absolute_discrimination,
            self.discrimination_index,
            self.novelty_preference,
            self.physical_object_set.object_bias_score[1],
        ]

    @property
    def all_physical_object_perimeters(self):
        return self.novel, self.familiar

    @cached_property
    def discrimination_index(self):
        return self.nort_absolute_discrimination / self.experiment_seconds

    @cached_property
    def novelty_preference(self):
        return 100.0 * self.physical_object_set["novel"].attention_filtered_seconds_observing / self.experiment_seconds

    @cached_property
    def nort_absolute_discrimination(self) -> float:
        """
        Definition: <frames observing novel object> - <frames observing constant object>

        :return:
        """
        try:
            return (
                self.physical_object_set["novel"].attention_filtered_seconds_observing
                - self.physical_object_set["constant"].attention_filtered_seconds_observing
            )
        except KeyError:
            msg = "The physical_objects must have a novel and a constant label " "to compute absolute_discrimination"
            raise AttributeError(msg)


class NortExperiment(RectanglePhysicalObjectExperiment):
    first_stage_has_no_object: ClassVar = True
    trial_classes: ClassVar = (
        NortHabituationTrial,
        NortTrainingTrial,
        NortNoveltyTrial,
    )
    experiment_stage_name_to_sequence_index: ClassVar[dict[str, int]] = {
        "habituation": 0,
        "open_field": 0,
        "training": 1,
        "1": 1,
        "t1": 1,
        "novelty": 2,
        "2": 2,
        "t2": 2,
        "test": 2,
        "novelty_observation": 2,
    }
