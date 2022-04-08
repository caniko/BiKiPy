from collections.abc import Sequence as AbcSequence
from logging import getLogger
from typing import Optional, Union, Sequence, ClassVar

from pydantic import Field, validator

from bikipy.behaviour.mixin.physical_object import PhysicalObjectExperimentMixin
from bikipy.behaviour.rectangle.square import SquareEnclosedExperiment, SquareEnclosedTrial
from bikipy.core.base_class import BikipyBase
from bikipy.core.typing import Perimeter2D
from bikipy.feature.physical_object import PhysicalObjectSet
from bikipy.perimeter.base import PerimeterSet
from bikipy.perimeter.polygon.base import PolygonPerimeter

logger = getLogger(__name__)


class ObjectField(BikipyBase):
    """
    The ObjectField class is used to store the spatial information of the perimeters across different stages
    of the object recognition trials.

    Each stage field can be inspected by using the item getter, <ObjectField object>[stage_id].
    """

    perimeters: dict[
        str, Union[dict[int, Perimeter2D], Sequence[Perimeter2D, ...], Perimeter2D]
    ] = Field(
        description=
        """
        Each perimeter type defined by the experiment design is a key-value pair, where the value is:
            - dict ->       The key is the stage index, and the value is the respective perimeter; useful when
                            the object is absent in some stages
            - sequence ->   The sequential definition of a perimeter, the perimeter must be available from stage 0,
                            and semantically speaking present till stage len(sequence)
            - perimeter ->  The perimeter is located in the same spatial coordinates across all the experiments
        """
    )

    @validator("perimeters")
    def ensure_label_definition(cls, value):
        for label, perimeter_reference in value:
            if isinstance(perimeter_reference, dict):
                for perimeter in perimeter_reference.values():
                    perimeter.label = label
            elif isinstance(perimeter_reference, AbcSequence):
                for perimeter in perimeter_reference:
                    perimeter.label = label
            elif isinstance(perimeter_reference, (PolygonPerimeter, PerimeterSet)):
                perimeter_reference.label = label
            else:
                msg = f"Invalid type in perimeters, {type(perimeter_reference)} -> {perimeter_reference}"
                raise ValueError(msg)

        return value

    def __getitem__(self, stage):
        if not isinstance(stage, int):
            msg = f"Only stage indexes are allowed, provide an integer; not {type(stage)} -> {stage}"
            raise ValueError(msg)
        if stage < 0:
            msg = f"Semantic error: Stage index must be a positive value, and not negative, {stage}"
            raise ValueError(msg)

        perimeter_set = []
        for label, perimeter_reference in self.perimeters.items():
            if isinstance(perimeter_reference, (dict, AbcSequence)):
                try:
                    perimeter = perimeter_reference[stage]
                except KeyError:
                    # The perimeter is not in this stage, skip the loop
                    continue
            elif isinstance(perimeter_reference, (PolygonPerimeter, PerimeterSet)):
                perimeter = perimeter_reference
            else:
                msg = f"Invalid type in perimeters, {type(perimeter_reference)} -> {perimeter_reference}"
                raise ValueError(msg)

            perimeter_set.append(perimeter)

        return perimeter_set

    def derive_physical_object_set(self, stage: int, **physical_object_set_kwargs):
        return PhysicalObjectSet.from_perimeter(*self[stage], **physical_object_set_kwargs)

    @classmethod
    def nort_format(
        cls,
        constant_object_perimeter: Perimeter2D,
        variable_object_perimeter: Perimeter2D,
        novel_object_perimeter: Perimeter2D,
        novelty_constant_object_perimeter: Optional[Perimeter2D] = None,
    ):
        return cls(
            perimeters={
                "novel": (variable_object_perimeter, novel_object_perimeter),
                "constant": (
                    constant_object_perimeter,
                    novelty_constant_object_perimeter,
                )
                if novelty_constant_object_perimeter
                else constant_object_perimeter,
            }
        )

    def nort_training_set(self, physical_object_set_kwargs) -> PhysicalObjectSet:
        # Hard-coding for backwards compatibility
        return self.derive_physical_object_set(*self[0], **physical_object_set_kwargs)

    def nort_novelty_set(self, physical_object_set_kwargs) -> PhysicalObjectSet:
        # Hard-coding for backwards compatibility
        return self.derive_physical_object_set(*self[1], **physical_object_set_kwargs)


class ObjectRecognitionExperiment(
    SquareEnclosedExperiment, PhysicalObjectExperimentMixin
):
    id_vs_object_field: Optional[dict[int, ObjectField]] = Field(
        None, description="Label of the nose in the df"
    )
    first_stage_has_no_object: ClassVar[bool] = True

    def trial_keyword_arguments(self, trial_id: int) -> dict:
        upstream_kwargs = super().trial_keyword_arguments(trial_id)

        if self.first_stage_has_no_object and upstream_kwargs["stage"] == 0:
            return upstream_kwargs

        return {
            **upstream_kwargs,
            **self._physical_object_keyword_arguments,
            "perimeter_border_normal_metric_magnitude": self.perimeter_border_normal_metric_magnitude,
            "nort_field": self.id_vs_object_field[upstream_kwargs["field_id"]],
        }


class ObjectRecognitionHabituationTrial(SquareEnclosedTrial):
    """The purpose of this stage is to generate reference data for proceeding experiments with objects."""
    trial_sequence_index: ClassVar[Optional[int]] = 0
    trial_label: ClassVar[str] = "Habituation"
