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

from bikipy.behaviour.mixin.physical_object import (
    RectanglePhysicalObjectExperiment,
    RectanglePhysicalObjectTrial,
)
from bikipy.perimeter.typing import AnyPerimeter


class ObjectsInUpdatingLocationsTrainingTrial(RectanglePhysicalObjectTrial):
    object_1: AnyPerimeter = ...
    object_2: AnyPerimeter = ...

    physical_object_labels: ClassVar[list[str, ...]] = ["object_1", "object_2"]

    experiment_sequence_index: ClassVar[int] = 0
    trial_label: ClassVar[str] = "Training"

    @property
    def all_physical_object_perimeters(self) -> tuple[AnyPerimeter, ...]:
        return self.object_1, self.object_2


class ObjectsInUpdatingLocationsUpdateTrial(RectanglePhysicalObjectTrial):
    object_1: AnyPerimeter = ...
    object_4: AnyPerimeter = ...

    physical_object_labels: ClassVar[list[str, ...]] = ["object_1", "object_4"]

    experiment_sequence_index: ClassVar[int] = 1
    trial_label: ClassVar[str] = "Update"

    @property
    def all_physical_object_perimeters(self) -> tuple[AnyPerimeter, ...]:
        return self.object_1, self.object_4


class ObjectsInUpdatingLocationsTestTrial(RectanglePhysicalObjectTrial):
    object_1: AnyPerimeter = ...
    object_2: AnyPerimeter = ...
    object_3: AnyPerimeter = ...
    object_4: AnyPerimeter = ...

    physical_object_labels: ClassVar[list[str, ...]] = ["object_1", "object_2", "object_3", "object_4"]

    experiment_sequence_index: ClassVar[int] = 2
    trial_label: ClassVar[str] = "Test"

    @property
    def all_physical_object_perimeters(self) -> tuple[AnyPerimeter, ...]:
        return self.object_1, self.object_2, self.object_3, self.object_4


class ObjectsInUpdatingLocationsExperiment(RectanglePhysicalObjectExperiment):
    trial_classes: ClassVar = (
        ObjectsInUpdatingLocationsTrainingTrial,
        ObjectsInUpdatingLocationsUpdateTrial,
        ObjectsInUpdatingLocationsTestTrial,
    )
