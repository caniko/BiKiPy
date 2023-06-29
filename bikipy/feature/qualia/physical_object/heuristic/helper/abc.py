from abc import ABC
from typing import TypeVar

from bikipy.feature.qualia.physical_object.heuristic.abc import AbstractHeuristic


class AbstractQualiaHelperHeuristic(AbstractHeuristic, ABC):
    heuristics_to_apply_to: tuple[str, ...] = ...


HelperHeuristic = TypeVar("HelperHeuristic", bound=AbstractQualiaHelperHeuristic)
