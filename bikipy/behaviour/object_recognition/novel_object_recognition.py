from functools import cached_property
from logging import getLogger
from typing import ClassVar

from bikipy.behaviour.mixin.physical_object import RectangleEnclosedPhysicalObjectTrial
from bikipy.behaviour.rectangle import RectangleEnclosedExperiment
from bikipy.perimeter.base import AnyPerimeter

logger = getLogger(__name__)


class NortTrainingTrial(RectangleEnclosedPhysicalObjectTrial):
    variable: AnyPerimeter
    familiar: AnyPerimeter

    physical_object_labels: ClassVar[list[str, ...]] = ["variable", "familiar"]

    experiment_class_name = "NortExperiment"
    trial_label: ClassVar[str] = "Training"

    @property
    def all_physical_object_perimeters(self):
        return self.variable, self.familiar


class NortNoveltyTrial(RectangleEnclosedPhysicalObjectTrial):
    novel: AnyPerimeter
    familiar: AnyPerimeter

    physical_object_labels: ClassVar[list[str, ...]] = ["novel", "familiar"]

    experiment_class_name = "NortExperiment"
    trial_label: ClassVar[str] = "Novelty"

    @classmethod
    @property
    def feature_headers(cls) -> list[tuple[str, ...]]:
        return super().feature_headers + [
            ("AbsoluteDiscrimination", "novel-familiar"),
            ("DiscriminationIndex", "novel-familiar"),
            ("NoveltyPreference", "novel-familiar"),
            ("ObjectBiasScore", "familiar"),
            ("ObjectBiasScore", "novel"),
        ]

    @property
    def feature_df_rows(self):
        return super().feature_df_rows + [
            self.nort_absolute_discrimination,
            self.discrimination_index,
            self.novelty_preference,
            *self.physical_object_set.object_bias_score.values(),
        ]

    @property
    def all_physical_object_perimeters(self):
        return self.novel, self.familiar

    @cached_property
    def discrimination_index(self):
        return self.nort_absolute_discrimination / self.physical_object_set.seconds_observing

    @cached_property
    def novelty_preference(self):
        return (
            100.0
            * self.physical_object_set["novel"].attention_filtered_seconds_observing
            / self.physical_object_set.seconds_observing
        )

    @cached_property
    def nort_absolute_discrimination(self) -> float:
        """
        Definition: <frames observing novel object> - <frames observing familiar object>

        :return:
        """
        try:
            return (
                self.physical_object_set["novel"].attention_filtered_seconds_observing
                - self.physical_object_set["familiar"].attention_filtered_seconds_observing
            )
        except KeyError:
            msg = "The physical_objects must have a novel and a familiar label " "to compute absolute_discrimination"
            raise AttributeError(msg)


class NortExperiment(RectangleEnclosedExperiment):
    first_stage_has_no_object: ClassVar = True
    trial_classes: ClassVar = (
        NortTrainingTrial,
        NortNoveltyTrial,
    )
    experiment_stage_name_to_stage_index: ClassVar[dict[str, int]] = {
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
