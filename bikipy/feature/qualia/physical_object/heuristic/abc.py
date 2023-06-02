from abc import ABC, abstractmethod
from typing import ClassVar, Type, TypeVar

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from pydantic import BaseModel, Field
from schemantic.model.project import SchemanticProjectMixin

from bikipy.core.video import VideoMetadataMixin
from bikipy.perimeter.base import SinglePerimeter
from bikipy.reader.base import Reader
from bikipy.utils.math.cached import cached_deg2rad, meters2pixels
from bikipy.utils.plot import TIGHT_LAYOUT_KWARGS


class AbstractQualiaHeuristic(VideoMetadataMixin, SchemanticProjectMixin, ABC):
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
        upstream = super().schemantic_fields_to_exclude_from_config_schema
        upstream.update(("perimeter", "reader"))
        return upstream

    @property
    @abstractmethod
    def result(self) -> np.ndarray[bool, bool]:
        ...

    @property
    @abstractmethod
    def summary_series(self) -> pd.Series:
        ...

    @abstractmethod
    def plot(self) -> None:
        ...

    @property
    def po_label(self) -> str:
        return self.perimeter.label

    def plot_result(self, ax: Axes, label_to_plot: str):
        ax.set_title("Combined")
        self.perimeter.plot(ax=ax, inspect_pixels=False)
        self.reader.plot_boolean_index(self.result, ax, label_to_plot)

        plt.tight_layout(**TIGHT_LAYOUT_KWARGS)


QualiaHeuristicCLS = Type[AbstractQualiaHeuristic]
QualiaHeuristic = TypeVar("QualiaHeuristic", bound=AbstractQualiaHeuristic)


class ProximityMixin(BaseModel):
    maximum_distance_meters: float = 0.05

    @property
    def maximum_distance_pixels(self) -> float:
        return meters2pixels(self.maximum_distance_meters, self.video.pixels_per_meter)


class RayMixin(BaseModel):
    maximum_degrees: float = 45.0

    @property
    def maximum_radians(self) -> float:
        return cached_deg2rad(self.maximum_degrees)
