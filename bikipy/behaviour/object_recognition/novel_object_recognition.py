from functools import cached_property
from logging import getLogger
from typing import ClassVar, Hashable

from bikipy.behaviour.mixin.physical_object import RectangleEnclosedPhysicalObjectTrial
from bikipy.behaviour.rectangle import RectangleEnclosedExperiment, RectangleEnclosedHabituationTrial
from bikipy.perimeter.base import SinglePerimeter

logger = getLogger(__name__)


class NortTrainingTrial(RectangleEnclosedPhysicalObjectTrial):
    variable: SinglePerimeter = ...
    familiar: SinglePerimeter = ...

    physical_object_labels = ("variable", "familiar")

    experiment_class_name = "NortExperiment"
    trial_label = "Training"

    @property
    def all_physical_object_perimeters(self):
        return self.variable, self.familiar


class NortNoveltyTrial(RectangleEnclosedPhysicalObjectTrial):
    novel: SinglePerimeter = ...
    familiar: SinglePerimeter = ...

    physical_object_labels = ("novel", "familiar")

    experiment_class_name = "NortExperiment"
    trial_label = "Novelty"

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
    habituation_trial_class = RectangleEnclosedHabituationTrial

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

    def trial_keyword_arguments(self, trial_id: Hashable) -> dict:
        upstream = super().trial_keyword_arguments(trial_id)

        match self.trial_id_to_trial_class_name[trial_id]:
            case "NortTrainingTrial":
                if "novel" in upstream:
                    logger.debug("Renaming perimeter label: novel to variable for use in NortTrainingTrial")

                    perimeter = upstream.pop("novel")
                    perimeter.label = "variable"
                    upstream["variable"] = upstream.pop("novel")

            case "NortNoveltyTrial":
                if "variable" in upstream:
                    logger.debug("Renaming perimeter label: variable to novel for use in NortNoveltyTrial")

                    perimeter = upstream.pop("variable")
                    perimeter.label = "novel"
                    upstream["novel"] = perimeter

        return upstream
