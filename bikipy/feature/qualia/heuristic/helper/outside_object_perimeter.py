from functools import cached_property
from typing import Optional

from pydantic import computed_field
from pydantic_numpy.typing import Np1DArrayBool

from bikipy.feature.qualia.axioms.proximity import ComputeProximity
from bikipy.feature.qualia.heuristic.helper.abc import (
    AbstractQualiaHelperHeuristic,
)
from bikipy.feature.qualia.heuristic.mixin import (
    ProximityMixin,
    SingleComponentMixin,
)


class OutsideObjectPerimeterHeuristic(SingleComponentMixin, ProximityMixin, AbstractQualiaHelperHeuristic):
    torso_label: str | None = "torso"

    manual_torso: Optional[ComputeProximity] = None

    heuristic_alias = "OutsideObjectPerimeter"

    @computed_field  # type: ignore[misc]
    @cached_property
    def result(self) -> Np1DArrayBool:
        if self.manual_torso is not None:
            return self.manual_torso

        if not self.torso_label:
            return None

        return ComputeProximity(
            label=self.heuristic_alias,
            perimeter=self.perimeter,
            maximum_distance=self.maximum_distance_pixels,
            inside_perimeter_border=self.reader[self.torso_label],
            manual_video=self.video,
        ).result

    @computed_field  # type: ignore[misc]
    @property
    def label_to_proximity_boolean(self) -> dict[str, Np1DArrayBool]:
        return {self.torso_label: self.result}
