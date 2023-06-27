from abc import ABC
from typing import ClassVar

from bikipy.feature.qualia.physical_object.heuristic.abc import AbstractQualiaHeuristic


class AbstractQualiaHelperHeuristic(AbstractQualiaHeuristic, ABC):
    must_be_true: ClassVar[bool] = ...
