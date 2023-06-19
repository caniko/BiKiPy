from functools import cached_property
from logging import getLogger
from typing import ClassVar

import pandas as pd

from bikipy.behaviour.core.constant import ExperimentStage
from bikipy.behaviour.core.enclosure.rectangle import (
    RectangleEnclosedExperiment,
    RectangleEnclosedHabituationTrial,
)
from bikipy.behaviour.object_recognition.generic import (
    RectangleEnclosedPhysicalObjectTrial,
)
from bikipy.core.typing import Label
from bikipy.perimeter.base import SinglePerimeter

logger = getLogger(__name__)


class NORTTrainingTrial(RectangleEnclosedPhysicalObjectTrial):
    variable: SinglePerimeter = ...
    familiar: SinglePerimeter = ...

    perimeter_labels = {"variable", "familiar"}

    experiment_class_name = "NORTExperiment"
    experiment_stage = ExperimentStage.TRAINING
    trial_label = "Familiarization"

    @classmethod
    @property
    def schemantic_fields_to_exclude_from_config_schema(cls) -> set[str]:
        upstream = super().schemantic_fields_to_exclude_from_config_schema
        upstream.update(("variable", "familiar"))
        return upstream

    @property
    def physical_object_perimeters(self) -> tuple[SinglePerimeter, ...]:
        return self.variable, self.familiar


class NORTNoveltyTrial(RectangleEnclosedPhysicalObjectTrial):
    novel: SinglePerimeter = ...
    familiar: SinglePerimeter = ...

    perimeter_labels = {"novel", "familiar"}

    experiment_class_name = "NORTExperiment"
    experiment_stage = ExperimentStage.TEST
    trial_label = "Novelty"

    @classmethod
    @property
    def schemantic_fields_to_exclude_from_config_schema(cls) -> set[str]:
        upstream = super().schemantic_fields_to_exclude_from_config_schema
        upstream.update(("novel", "familiar"))
        return upstream

    @cached_property
    def _trial_physical_object_feature_series_list(self) -> list[pd.Series]:
        upstream_list = super()._trial_physical_object_feature_series_list
        upstream_list.append(
            pd.Series(
                (
                    self.nort_absolute_discrimination,
                    self.discrimination_index,
                    self.novelty_preference,
                ),
                index=(
                    ("AbsoluteDiscrimination", "NovelFamiliar"),
                    ("DiscriminationIndex", "NovelFamiliar"),
                    ("NoveltyPreference", "NovelFamiliar"),
                ),
            )
        )
        return upstream_list

    @property
    def physical_object_perimeters(self) -> tuple[SinglePerimeter, ...]:
        return self.novel, self.familiar

    @cached_property
    def discrimination_index(self):
        return self.nort_absolute_discrimination / self.physical_object_set.po_total_seconds_observing

    @cached_property
    def novelty_preference(self):
        return (
            100.0
            * self.physical_object_set["novel"].attention_filtered_seconds_observing
            / self.physical_object_set.po_total_seconds_observing
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


class NORTExperiment(RectangleEnclosedExperiment):
    experiment_labels = {"nort", "novel_object_recognition_test"}

    habituation_trial_class = RectangleEnclosedHabituationTrial
    trial_sequence = (NORTTrainingTrial, NORTNoveltyTrial)

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

    def trial_keyword_arguments(self, trial_id: Label) -> dict:
        upstream = super().trial_keyword_arguments(trial_id)

        match self.trial_id_to_trial_class_name[trial_id]:
            case "NORTTrainingTrial":
                if "novel" in upstream:
                    logger.debug("Renaming perimeter label: novel to variable for use in NORTTrainingTrial")

                    perimeter = upstream.pop("novel")
                    perimeter.label = "variable"
                    upstream["variable"] = upstream.pop("novel")

            case "NORTNoveltyTrial":
                if "variable" in upstream:
                    logger.debug("Renaming perimeter label: variable to novel for use in NORTNoveltyTrial")

                    perimeter = upstream.pop("variable")
                    perimeter.label = "novel"
                    upstream["novel"] = perimeter

        return upstream
