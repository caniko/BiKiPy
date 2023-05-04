from abc import ABC, abstractmethod
from typing import TypeVar, Optional, ClassVar, Type

import pandas as pd
from matplotlib.axes import Axes
from pydantic import Field, BaseModel
from pydantic_numpy import NDArrayBool
from schemantic.model.project import SchemanticProjectMixin

from bikipy.core.video import VideoMetadataMixin
from bikipy.perimeter.base import SinglePerimeter
from bikipy.reader.base import Reader
from bikipy.utils.math.cached import meters2pixels, cached_deg2rad


class AbstractQualiaProfile(VideoMetadataMixin, SchemanticProjectMixin, ABC):
    perimeter: SinglePerimeter = ...
    reader: Reader = ...
    filter_in_sequence: bool = Field(
        False,
        description="When set to True, the component boolean index will be "
        "considered in sequence with other components that are also filtered in sequence",
    )

    label: ClassVar[str] = ...

    @property
    @abstractmethod
    def result(self) -> NDArrayBool:
        ...

    @property
    @abstractmethod
    def summary_series(self) -> pd.Series:
        ...

    @abstractmethod
    def plot(self, ax: Optional[Axes] = None) -> None:
        ...

    def plot_result(self, ax: Axes) -> None:
        ax.set_title(self.label)
        self.reader.plot_boolean_index(self.result, ax)


QualiaProfileCLS = Type[AbstractQualiaProfile]
QualiaProfile = TypeVar("QualiaProfile", bound=AbstractQualiaProfile)


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
