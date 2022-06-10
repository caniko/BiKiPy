from collections.abc import Sequence as AbcSequence
from functools import cached_property
from typing import Optional, Sequence

from pydantic import Field, validator

from bikipy.core.base_class import BikipyBase
from bikipy.perimeter.typing import AnyPerimeter
from bikipy.feature.physical_object.core import PhysicalObjectSet
from bikipy.perimeter.base import PerimeterSet
from bikipy.perimeter.polygon.base import PolygonPerimeter


class ObjectField(BikipyBase):
    """
    The ObjectField class is used to store the spatial information of the perimeters across different stages
    of the object recognition trials.

    Each stage field can be inspected by using the item getter, <ObjectField object>[stage_id].
    """

    object_field_perimeter_sequence: (
        AnyPerimeter
        | Sequence[AnyPerimeter]
        | dict[int, AnyPerimeter]
        | dict[str, AnyPerimeter]
        | dict[str, Sequence[AnyPerimeter]]
        | dict[str, dict[int, AnyPerimeter]]
    ) = Field(
        ...,
        description="""
        Each perimeter type defined by the experiment design is a key-value pair, where the value is:
            - sequence ->   The sequence is based on stages; for instance, the perimeter on index 0 belongs to stage 0.
                            Must have the same length as the number of stages.
            - dict[int] ->  The key is the stage index, and the value is the respective perimeter; useful when
                            the object is absent in some stages
            - dict[str] ->  AUTHOR'S CHOICE. The key is the perimeter label, and the value can be dict[int] or sequence.
            - perimeter ->  The perimeter is located in the same spatial coordinates across all the experiments
        """,
    )

    @classmethod
    def __modify_schema__(cls, field_schema):
        # __modify_schema__ should mutate the dict it receives in place,
        # the returned value will be ignored
        field_schema.update({"type": "bikipy.feature.physical_object.ObjectField"})

    def __len__(self):
        return len(self.object_field_perimeter_sequence)

    def __getitem__(self, stage):
        if not isinstance(stage, int):
            msg = f"Only stage indexes are allowed, provide an integer; not {type(stage)} -> {stage}"
            raise ValueError(msg)
        if stage < 0:
            msg = f"Semantic error: Stage index must be a positive value, and not negative, {stage}"
            raise ValueError(msg)

        perimeter_set = []
        for label, perimeter_reference in self.object_field_perimeter_sequence.items():
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

    @validator("object_field_perimeter_sequence")
    def ensure_label_definition(cls, value):
        for label, perimeter_reference in value.items():
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

    @cached_property
    def perimeter_labels(self) -> tuple[str]:
        return tuple(self.object_field_perimeter_sequence)

    def derive_physical_object_set(self, stage: int, **physical_object_set_kwargs):
        return PhysicalObjectSet.from_perimeter(*self[stage], **physical_object_set_kwargs)

    @classmethod
    def from_perimeter_set(cls, perimeter_set: PerimeterSet, with_restricted: bool = False):
        label_to_perimeter = {}
        for perimeter in perimeter_set.all_perimeters if with_restricted else perimeter_set.perimeters:
            if not perimeter.label:
                msg = "A perimeter in the provided PerimeterSet has no label"
                raise ValueError(msg)
            if perimeter.label in label_to_perimeter:
                msg = "Duplicate perimeter labels"
                raise ValueError(msg)
            label_to_perimeter[perimeter.label] = perimeter
        return cls(object_field_perimeter_sequence=label_to_perimeter)

    @classmethod
    def nort_format(
        cls,
        constant_object_perimeter: AnyPerimeter,
        variable_object_perimeter: AnyPerimeter,
        novel_object_perimeter: AnyPerimeter,
        novelty_constant_object_perimeter: Optional[AnyPerimeter] = None,
    ):
        return cls(
            object_field_perimeter_sequence={
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
