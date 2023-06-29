from abc import ABC
from typing import Optional, TypeVar

from matplotlib.axes import Axes

from bikipy.feature.qualia.physical_object.heuristic.abc import StandaloneHeuristic


class AbstractQualiaHelperHeuristic(StandaloneHeuristic, ABC):
    def plot_result(self, ax: Axes, label_to_plot: Optional[str] = None) -> None:
        ax.set_title(self.heuristic_alias)
        super().plot_result(ax, label_to_plot)


HelperHeuristic = TypeVar("HelperHeuristic", bound=AbstractQualiaHelperHeuristic)
