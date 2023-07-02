from abc import ABC, abstractmethod
from typing import ClassVar, Optional, Type, TypeVar

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from pydantic import Field
from pydantic_numpy import NDArrayBool
from schemantic.model.project import SchemanticProjectMixin

from bikipy.core.video import VideoMetadataMixin
from bikipy.feature.qualia.physical_object.heuristic.mixin import SingleComponentMixin
from bikipy.perimeter.base import SinglePerimeter, Perimeter
from bikipy.reader.base import Reader
from bikipy.utils.plot import TIGHT_LAYOUT_KWARGS


class AbstractHeuristic(VideoMetadataMixin, SchemanticProjectMixin, ABC):
    perimeter: SinglePerimeter = ...
    reader: Reader = ...

    filter_in_sequence: bool = Field(
        False,
        description="When set to True, the component boolean index will be "
        "considered in sequence with other components that are also filtered in sequence",
    )

    heuristic_alias: ClassVar[str]

    @classmethod
    @property
    def schemantic_fields_to_exclude_from_config_schema(cls) -> set[str]:
        result = super().schemantic_fields_to_exclude_from_config_schema
        result.update(("perimeter", "reader"))
        return result

    @abstractmethod
    def plot(self) -> None:
        ...

    @property
    def perimeter_to_boolean_index(self) -> dict[Perimeter, np.ndarray[bool, bool]]:
        return {self.perimeter: self.result}

    @property
    def physical_object_label(self) -> str:
        return self.perimeter.label

    def plot_result(self, ax: Axes, label_to_plot: Optional[str] = None) -> None:
        self.perimeter.plot(ax=ax, coordinates_as_pixels=False)
        self.reader.plot_boolean_index(self.result, ax, label_to_plot)

        plt.tight_layout(**TIGHT_LAYOUT_KWARGS)


HeuristicCLS = Type[AbstractHeuristic]
Heuristic = TypeVar("Heuristic", bound=AbstractHeuristic)


class StandaloneHeuristic(AbstractHeuristic):
    @property
    @abstractmethod
    def result(self) -> np.ndarray[bool, bool]:
        ...

    def plot_result(self, ax: Axes, label_to_plot: Optional[str] = None) -> None:
        ax.set_title("Combined result")
        super().plot_result(ax, label_to_plot)

    @property
    def summary_series(self) -> pd.Series:
        return pd.Series(
            [self.video.boolean_array_to_seconds(self.result)],
            index=[f"{self.perimeter.label.capitalize()}{self.heuristic_alias}"],
        )


class CombinedHeuristic(SingleComponentMixin, AbstractHeuristic):
    label: str = ...
    result: NDArrayBool = ...

    @property
    def summary_series(self) -> pd.Series:
        return pd.Series([self.video.boolean_array_to_seconds(self.result)], index=[self.label])
