from typing import ClassVar

from bikipy.feature.qualia.physical_object.heuristic.abc import AbstractQualiaHeuristic


class WhiskerInteractionQualiaGroup(AbstractQualiaHeuristic):
    whisker_midpoint_distance_from_nose_to_eye: ClassVar[float] = 0.2
    left_whisker_label: ClassVar[str] = "left_whisker"
    right_whisker_label: ClassVar[str] = "right_whisker"

    @property
    def result(self) -> np.ndarray[bool, bool]:
        pass
