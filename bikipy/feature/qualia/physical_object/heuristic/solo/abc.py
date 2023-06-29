from abc import ABC, abstractmethod
from functools import cached_property
from typing import Iterable, Optional

import numpy as np
from matplotlib.axes import Axes
from pydantic_numpy import NDArrayBool

from bikipy.feature.qualia.physical_object.heuristic.abc import StandaloneHeuristic
from bikipy.feature.qualia.physical_object.heuristic.helper.abc import HelperHeuristic


class AbstractSoloHeuristic(StandaloneHeuristic, ABC):
    combined_helper_heuristic: Optional[NDArrayBool]

    @property
    @abstractmethod
    def solo_result(self) -> np.ndarray[bool, bool]:
        ...

    @cached_property
    def result(self) -> np.ndarray[bool, bool]:
        return self._apply_helper_heuristics(self.solo_result)

    def _apply_helper_heuristics(self, target: np.ndarray[bool, bool]) -> np.ndarray[bool, bool]:
        if hasattr(self, "combined_helper_heuristic"):
            return target & self.combined_helper_heuristic
        return target

    def _plot_helper_heuristics(self, axes: Iterable[Axes]) -> None:
        for ax, helper_heuristic in zip(axes, self.helper_heuristics):
            helper_heuristic.plot_result(ax)
