from abc import ABC, abstractmethod
from typing import ClassVar, Optional

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.axes import Axes
from pydantic import Field, computed_field
from pydantic_numpy.typing import Np1DArrayBool
from schemantic import SchemanticProjectModelMixin

from bikipy.core.video import VideoMetadataMixin
from bikipy.feature.qualia.heuristic.mixin import SingleComponentMixin
from bikipy.perimeter.base import BasePerimeter, BaseSinglePerimeter
from bikipy.plot import TIGHT_LAYOUT_KWARGS
from bikipy.reader.base import BaseReader


class AbstractHeuristic(VideoMetadataMixin, SchemanticProjectModelMixin, ABC):
    perimeter: BaseSinglePerimeter
    reader: BaseReader

    filter_in_sequence: bool = Field(
        False,
        description="When set to True, the component boolean index will be "
        "considered in sequence with other components that are also filtered in sequence",
    )

    heuristic_alias: ClassVar[str]

    @classmethod
    @property
    def fields_to_exclude_from_single_schema(cls) -> set[str]:
        result = super().fields_to_exclude_from_single_schema
        result.update(("perimeter", "reader"))
        return result

    @abstractmethod
    def plot(self) -> plt.Figure: ...

    @computed_field  # type: ignore[misc]
    @property
    def perimeter_to_boolean_index(self) -> dict[BasePerimeter, Np1DArrayBool]:
        return {self.perimeter: self.result}

    @computed_field  # type: ignore[misc]
    @property
    def physical_object_label(self) -> str:
        return self.perimeter.label

    def plot_result(self, ax: Axes, label_to_plot: Optional[str] = None) -> None:
        self.perimeter.plot_perimeter_on_ax(ax=ax, coordinates_as_pixels=False)
        self.reader.plot_boolean_index(self.result, ax, label_to_plot)

        plt.tight_layout(**TIGHT_LAYOUT_KWARGS)


HeuristicCLS = type[AbstractHeuristic]


class StandaloneHeuristic(AbstractHeuristic):
    @property
    @abstractmethod
    def result(self) -> Np1DArrayBool: ...

    def plot_result(self, ax: Axes, label_to_plot: Optional[str] = None) -> None:
        ax.set_title("Combined result")
        super().plot_result(ax, label_to_plot)

    @computed_field  # type: ignore[misc]
    @property
    def summary_series(self) -> pd.Series:
        return pd.Series(
            [self.video.boolean_array_to_seconds(self.result)],
            index=[f"{self.perimeter.label.capitalize()}{self.heuristic_alias}"],
        )


class CombinedHeuristic(SingleComponentMixin, AbstractHeuristic):
    label: str
    result: Np1DArrayBool

    @computed_field  # type: ignore[misc]
    @property
    def summary_series(self) -> pd.Series:
        return pd.Series([self.video.boolean_array_to_seconds(self.result)], index=[self.label])
