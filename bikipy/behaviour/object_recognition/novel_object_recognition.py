from functools import cached_property
from logging import getLogger
from typing import ClassVar

import pandas as pd
from pydantic import computed_field

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

    @computed_field(return_type=set[str])
    @classmethod
    @property
    def schemantic_fields_to_exclude_from_config_schema(cls) -> set[str]:
        result = super().schemantic_fields_to_exclude_from_config_schema
        result.update(("variable", "familiar"))
        return result

    @computed_field
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

    @computed_field(return_type=set[str])
    @classmethod
    @property
    def schemantic_fields_to_exclude_from_config_schema(cls) -> set[str]:
        result = super().schemantic_fields_to_exclude_from_config_schema
        result.update(("novel", "familiar"))
        return result

    @computed_field
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

    @computed_field
    @property
    def physical_object_perimeters(self) -> tuple[SinglePerimeter, ...]:
        return self.novel, self.familiar

    @computed_field
    @cached_property
    def discrimination_index(self):
        return self.nort_absolute_discrimination / self.physical_object_set.po_total_seconds_observing

    @computed_field
    @cached_property
    def novelty_preference(self):
        return (
            100.0
            * self.physical_object_set["novel"].attention_filtered_seconds_observing
            / self.physical_object_set.po_total_seconds_observing
        )

    @computed_field
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
        result = super().trial_keyword_arguments(trial_id)

        match self.trial_id_to_trial_class_name[trial_id]:
            case "NORTTrainingTrial":
                if "novel" in result:
                    logger.debug("Renaming perimeter label: novel to variable for use in NORTTrainingTrial")

                    perimeter = result.pop("novel")
                    perimeter.label = "variable"
                    result["variable"] = result.pop("novel")

            case "NORTNoveltyTrial":
                if "variable" in result:
                    logger.debug("Renaming perimeter label: variable to novel for use in NORTNoveltyTrial")

                    perimeter = result.pop("variable")
                    perimeter.label = "novel"
                    result["novel"] = perimeter

        return result
