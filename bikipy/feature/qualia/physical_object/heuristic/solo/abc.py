from abc import ABC
from functools import cached_property
from typing import Iterable

import numpy as np
from matplotlib.axes import Axes

from bikipy.feature.qualia.physical_object.heuristic.abc import StandaloneHeuristic
from bikipy.feature.qualia.physical_object.heuristic.helper.abc import HelperHeuristic


class AbstractSoloHeuristic(StandaloneHeuristic, ABC):
    combined_helper_heuristic: np.ndarray[bool, bool]

    def _plot_helper_heuristics(self, axes: Iterable[Axes]) -> None:
        for ax, helper_heuristic in zip(axes, self.helper_heuristics):
            helper_heuristic.plot_result(ax)
