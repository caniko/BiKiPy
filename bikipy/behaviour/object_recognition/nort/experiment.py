from dataclasses import dataclass, field
from logging import getLogger
from typing import Optional, Union

from pydantic import Field

from bikipy.behaviour.mixin.physical_object import PhysicalObjectExperimentMixin
from bikipy.behaviour.object_recognition.nort.constants import TRIAL_LABEL_VS_CLASS_NAME
from bikipy.behaviour.square import SquareEnclosedExperiment
from bikipy.core.typing import Perimeter2D
from bikipy.feature.physical_object import PhysicalObjectSet
from bikipy.perimeter.base import distance_between_two_perimeters

logger = getLogger(__name__)


class NortExperiment(SquareEnclosedExperiment, PhysicalObjectExperimentMixin):
    nort_field_id_vs_nort_field_object: Optional[dict] = Field(
        None, description="Label of the nose in the df"
    )

    def trial_keyword_arguments(self, trial_id: int) -> dict:
        result = super().trial_keyword_arguments(trial_id)

        if TRIAL_LABEL_VS_CLASS_NAME[result["stage"]] == "habituation":
            return result

        return {
            **result,
            **self._physical_object_keyword_arguments,
            "perimeter_border_normal_metric_magnitude": self.perimeter_border_normal_metric_magnitude,
            "nort_field": self.nort_field_id_vs_nort_field_object[result["field_id"]],
        }


@dataclass(frozen=True, order=True)
class NortField:
    label: int
    constant_object_perimeter: Perimeter2D
    variable_object_perimeter: Perimeter2D
    novel_object_perimeter: Perimeter2D
    novelty_constant_object_perimeter: Optional[Perimeter2D] = None

    def __post_init__(self):
        if self.novelty_constant_object_perimeter:
            self.constant_object_perimeter.label = "training_constant"
            self.novelty_constant_object_perimeter.label = "novel_constant"
        else:
            self.constant_object_perimeter.label = "constant"

        self.variable_object_perimeter.label = "variable"
        self.novel_object_perimeter.label = "novel"

    @classmethod
    def from_undefined(
        cls,
        label: int,
        habituation_object_perimeter_a: Perimeter2D,
        habituation_object_perimeter_b: Perimeter2D,
        novel_object_perimeter: Perimeter2D,
    ):
        if distance_between_two_perimeters(
            habituation_object_perimeter_a, novel_object_perimeter
        ) < distance_between_two_perimeters(
            habituation_object_perimeter_b, novel_object_perimeter
        ):
            constant_object_perimeter = habituation_object_perimeter_b
            variable_object_perimeter = habituation_object_perimeter_a
        else:
            constant_object_perimeter = habituation_object_perimeter_a
            variable_object_perimeter = habituation_object_perimeter_b

        return cls(
            label,
            constant_object_perimeter,
            variable_object_perimeter,
            novel_object_perimeter,
        )

    def __getitem__(self, item: str):
        if not isinstance(item, str):
            msg = f"{self.__class__.__name__} only accepts string for getting item"
            raise TypeError(msg)
        if TRIAL_LABEL_VS_CLASS_NAME[item] == "training":
            return self.training_set
        if TRIAL_LABEL_VS_CLASS_NAME[item] == "novelty":
            return self.novelty_set

        if TRIAL_LABEL_VS_CLASS_NAME[item] == "habituation":
            msg = "habituation has no physical objects."
        else:
            msg = f"{item} is neither training nor novelty related."
        raise ValueError(msg)

    def training_set(self, physical_object_set_kwargs) -> PhysicalObjectSet:
        return PhysicalObjectSet.from_perimeter(
            self.variable_object_perimeter,
            self.constant_object_perimeter,
            **physical_object_set_kwargs,
        )

    def novelty_set(self, physical_object_set_kwargs) -> PhysicalObjectSet:
        return PhysicalObjectSet.from_perimeter(
            self.novel_object_perimeter,
            self.novelty_constant_object_perimeter or self.constant_object_perimeter,
            **physical_object_set_kwargs,
        )
