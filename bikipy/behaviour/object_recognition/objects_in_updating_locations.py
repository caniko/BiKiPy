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
WT: 3 > 4 > 1 > 2
"""
from typing import ClassVar

from bikipy.behaviour.object_recognition.base import (
    GenericObjectRecognitionTrial,
    ObjectRecognitionExperiment,
)
from bikipy.feature.physical_object.field import ObjectField
from bikipy.perimeter.typing import AnyPerimeter


def _four_objects_are_indexes(object_field: ObjectField):
    if any(i not in object_field.perimeters for i in (1, 2, 3, 4)):
        msg = (
            f"The object_field does not define the required labels for the experiment: {object_field.perimeters.keys()}"
        )
        raise ValueError(msg)


class ObjectsInUpdatingLocationsTrainingTrial(GenericObjectRecognitionTrial):
    object_1: AnyPerimeter = ...
    object_2: AnyPerimeter = ...

    trial_stage_index: ClassVar[int] = 0
    trial_label: ClassVar[str] = "Training"

    @property
    def all_physical_object_perimeters(self) -> tuple[AnyPerimeter, ...]:
        return self.object_1, self.object_2


class ObjectsInUpdatingLocationsUpdateTrial(GenericObjectRecognitionTrial):
    object_1: AnyPerimeter = ...
    object_4: AnyPerimeter = ...

    trial_stage_index: ClassVar[int] = 1
    trial_label: ClassVar[str] = "Update"

    @property
    def all_physical_object_perimeters(self) -> tuple[AnyPerimeter, ...]:
        return self.object_1, self.object_4


class ObjectsInUpdatingLocationsTestTrial(GenericObjectRecognitionTrial):
    object_1: AnyPerimeter = ...
    object_2: AnyPerimeter = ...
    object_3: AnyPerimeter = ...
    object_4: AnyPerimeter = ...

    trial_stage_index: ClassVar[int] = 2
    trial_label: ClassVar[str] = "Test"

    @property
    def all_physical_object_perimeters(self) -> tuple[AnyPerimeter, ...]:
        return self.object_1, self.object_2, self.object_3, self.object_4


class ObjectsInUpdatingLocationsExperiment(ObjectRecognitionExperiment):
    trial_classes: ClassVar = (
        ObjectsInUpdatingLocationsTrainingTrial,
        ObjectsInUpdatingLocationsUpdateTrial,
        ObjectsInUpdatingLocationsTestTrial,
    )
