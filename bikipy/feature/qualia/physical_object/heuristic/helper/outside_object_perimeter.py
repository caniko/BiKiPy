from functools import cached_property
from typing import Optional

import numpy as np

from bikipy.feature.qualia.axioms.proximity import ComputeProximity
from bikipy.feature.qualia.physical_object.heuristic.abc import ProximityMixin
from bikipy.feature.qualia.physical_object.heuristic.helper.abc import (
    AbstractQualiaHelperHeuristic,
)


class OutsideObjectPerimeterHeuristic(AbstractQualiaHelperHeuristic, ProximityMixin):
    torso_label: str | None = "torso"

    manual_torso: Optional[ComputeProximity]

    heuristic_alias = "TorsoOutsideObjectPerimeter"

    @cached_property
    def result(self) -> np.ndarray[bool, bool]:
        if self.manual_torso is not None:
            return self.manual_torso

        if not self.torso_label:
            return None

        return ComputeProximity(
            perimeter=self.perimeter,
            maximum_distance=self.maximum_distance_pixels,
            inside_perimeter_border=self.reader[self.torso_label],
            label="Torso",
            manual_video=self.video,
        )

    @property
    def label_to_proximity_boolean(self) -> dict[str, np.ndarray[bool, bool]]:
        return {self.torso_label: self.result}
