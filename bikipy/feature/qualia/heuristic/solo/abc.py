from abc import ABC, abstractmethod
from functools import cached_property
from typing import Iterable, Optional

from matplotlib.axes import Axes
from pydantic import computed_field
from pydantic_numpy.typing import Np1DArrayBool

from bikipy.feature.qualia.heuristic.abc import StandaloneHeuristic


class AbstractSoloHeuristic(StandaloneHeuristic, ABC):
    combined_helper_heuristic: Optional[Np1DArrayBool] = None

    @property
    @abstractmethod
    def solo_result(self) -> Np1DArrayBool: ...

    @computed_field  # type: ignore[misc]
    @cached_property
    def result(self) -> Np1DArrayBool:
        return self._apply_helper_heuristics(self.solo_result)

    def _apply_helper_heuristics(self, target: Np1DArrayBool) -> Np1DArrayBool:
        if self.combined_helper_heuristic:
            return target & self.combined_helper_heuristic
        return target

    def _plot_helper_heuristics(self, axes: Iterable[Axes]) -> None:
        for ax, helper_heuristic in zip(axes, self.helper_heuristics):
            helper_heuristic.plot_result(ax)
