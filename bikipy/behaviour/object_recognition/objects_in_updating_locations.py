"""
This experiment includes four objects. These objects are not present in every stage. The index enumeration
is from left to right and from top to bottom:
    - Stage 1 (Training): Object 1 and 2 are present
    - Stage 2 (Update): Object 1 and 4
    - Stage 3 (Test): All 4 objects

Object 1, present in stages, is a positive control for memory
Object 2, present in all but stage 2 - Update, is a positive control for long-term memory
Object 3, present in stage 3 - Test, is a novel object and previously uncontested position
Object 4, present in stage 2 and 3, positive control for updated memory

Test hypothesis:
TG: Equal
WT: 4 > 3 > 2 >~ 1
"""
from pydantic.main import ModelMetaclass

from bikipy.behaviour.core.enclosure.rectangle import (
    RectangleEnclosedExperiment,
    RectangleEnclosedHabituationTrial,
)
from bikipy.behaviour.object_recognition.generic import (
    RectangleEnclosedPhysicalObjectTrial,
)
from bikipy.perimeter.base import SinglePerimeter


class ObjectsInUpdatingLocationsTrainingTrial(RectangleEnclosedPhysicalObjectTrial):
    object_1: SinglePerimeter = ...
    object_2: SinglePerimeter = ...

    perimeter_labels = {"object_1", "object_2"}

    experiment_class_name = "ObjectsInUpdatingLocationsExperiment"
    trial_label = "Training"

    @classmethod
    @property
    def schemantic_fields_to_exclude_from_config_schema(cls) -> set[str]:
        upstream = super().schemantic_fields_to_exclude_from_config_schema
        upstream.update(("object_1", "object_2"))
        return upstream

    @property
    def physical_object_perimeters(self) -> tuple[SinglePerimeter, ...]:
        return self.object_1, self.object_2


class ObjectsInUpdatingLocationsUpdateTrial(RectangleEnclosedPhysicalObjectTrial):
    object_1: SinglePerimeter = ...
    object_3: SinglePerimeter = ...

    perimeter_labels = {"object_1", "object_3"}

    experiment_class_name = "ObjectsInUpdatingLocationsExperiment"
    trial_label = "Update"

    @classmethod
    @property
    def schemantic_fields_to_exclude_from_config_schema(cls) -> set[str]:
        upstream = super().schemantic_fields_to_exclude_from_config_schema
        upstream.update(("object_1", "object_3"))
        return upstream

    @property
    def physical_object_perimeters(self) -> tuple[SinglePerimeter, ...]:
        return self.object_1, self.object_3


class ObjectsInUpdatingLocationsTestTrial(RectangleEnclosedPhysicalObjectTrial):
    object_1: SinglePerimeter = ...
    object_2: SinglePerimeter = ...
    object_3: SinglePerimeter = ...
    object_4: SinglePerimeter = ...

    perimeter_labels = {"object_1", "object_2", "object_3", "object_4"}

    experiment_class_name = "ObjectsInUpdatingLocationsExperiment"
    trial_label = "Test"

    @classmethod
    @property
    def schemantic_fields_to_exclude_from_config_schema(cls) -> set[str]:
        upstream = super().schemantic_fields_to_exclude_from_config_schema
        upstream.update(("object_1", "object_2", "object_3", "object_4"))
        return upstream

    @property
    def physical_object_perimeters(self) -> tuple[SinglePerimeter, ...]:
        return self.object_1, self.object_2, self.object_3, self.object_4

    # @cached_property
    # def _trial_physical_object_feature_series_list(self) -> list[pd.Series]:
    #     base = super()._trial_physical_object_feature_series_list
    #     return base


class ObjectsInUpdatingLocationsExperiment(RectangleEnclosedExperiment):
    experiment_labels = {"oul", "objects_in_updating_locations"}

    habituation_trial_class = RectangleEnclosedHabituationTrial
    trial_sequence = (
        ObjectsInUpdatingLocationsTrainingTrial,
        ObjectsInUpdatingLocationsUpdateTrial,
        ObjectsInUpdatingLocationsTestTrial,
    )
