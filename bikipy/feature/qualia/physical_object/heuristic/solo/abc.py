from abc import ABC, abstractmethod
from typing import Iterable, TypeVar

from matplotlib.axes import Axes

from bikipy.feature.qualia.physical_object.heuristic.abc import AbstractHeuristic
from bikipy.feature.qualia.physical_object.heuristic.helper.abc import HelperHeuristic


class AbstractSoloHeuristic(AbstractHeuristic, ABC):
    helper_heuristics: list[HelperHeuristic]

    @abstractmethod
    def plot(self) -> None:
        ...

    def _plot_helper_heuristics(self, axes: Iterable[Axes]) -> None:
        for ax, helper_heuristic in zip(axes, self.helper_heuristics):
            helper_heuristic.pl


SoloHeuristic = TypeVar("SoloHeuristics", bound=AbstractSoloHeuristic)
