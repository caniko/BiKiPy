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
from typing import Any, ClassVar

from pydantic import root_validator, validator

from bikipy.behaviour.object_recognition.base import (
    GenericObjectRecognitionTrial,
    ObjectRecognitionExperiment,
)
from bikipy.feature.physical_object.field import ObjectField


def _four_objects_are_indexes(object_field: ObjectField):
    if any(i not in object_field.perimeters for i in (1, 2, 3, 4)):
        msg = (
            f"The object_field does not define the required labels for the experiment: {object_field.perimeters.keys()}"
        )
        raise ValueError(msg)


class ObjectUpdateTaskTraining(GenericObjectRecognitionTrial):
    trial_stage_index: ClassVar[int] = 0
    trial_label: ClassVar[str] = "Training"

    @validator("object_field")
    def object_field_defined_1_and_2(cls, value):
        if 1 not in value.perimeters or 2 not in value.perimeters:
            msg = f"1 and 2 has to be defined in object_field when assigned to {cls.__name__}"
            raise AttributeError(msg)
        return value


class ObjectUpdateTaskUpdate(GenericObjectRecognitionTrial):
    trial_stage_index: ClassVar[int] = 1
    trial_label: ClassVar[str] = "Update"

    @validator("object_field")
    def object_field_defined_1_and_4(cls, value):
        if 1 not in value.perimeters or 4 not in value.perimeters:
            msg = f"1 and 4 has to be defined in object_field when assigned to {cls.__name__}"
            raise AttributeError(msg)
        return value


class ObjectUpdateTaskTest(GenericObjectRecognitionTrial):
    trial_stage_index: ClassVar[int] = 2
    trial_label: ClassVar[str] = "Test"

    @validator("object_field")
    def object_field_defined_all_4(cls, value):
        if _four_objects_are_indexes(value):
            msg = f"All 4 objects has to be defined in object_field when assigned to {cls.__name__}"
            raise AttributeError(msg)
        return value


class ObjectUpdateTaskExperiment(ObjectRecognitionExperiment):
    trial_classes: ClassVar = (
        ObjectUpdateTaskTraining,
        ObjectUpdateTaskUpdate,
        ObjectUpdateTaskTest,
    )

    @root_validator
    def four_objects_must_be_defined_and_are_indexes(cls, values):
        if "global_object" in values:
            _four_objects_are_indexes(values["global_object"])
        else:
            for object_field in values["id_vs_object_field"].values():
                _four_objects_are_indexes(object_field)
        return values
