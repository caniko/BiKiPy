from typing import ClassVar

from pydantic_numpy import NDArrayBool

from bikipy.feature.physical_object.component.abc import AbcQualiaComponent


class WhiskerRayCast(AbcQualiaComponent):
    whisker_midpoint_distance_from_nose_to_eye: ClassVar[float] = 0.2
    left_whisker_label: ClassVar[str] = "left_whisker"
    right_whisker_label: ClassVar[str] = "right_whisker"

    @property
    def boolean_index(self) -> NDArrayBool:
        pass
