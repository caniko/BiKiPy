from abc import ABC, abstractmethod
from typing import TypeVar, Optional

import pandas as pd
from matplotlib.axes import Axes
from pydantic import Field
from pydantic_numpy import NDArrayBool

from bikipy.core.video import VideoMetadataMixin
from bikipy.perimeter.base import SinglePerimeter
from bikipy.reader.base import Reader


class AbstractQualiaComponent(VideoMetadataMixin, ABC):
    perimeter: SinglePerimeter = ...
    reader: Reader = ...
    filter_in_sequence: bool = Field(
        False,
        description="When set to True, the component boolean index will be "
        "considered in sequence with other components that are also filtered in sequence",
    )

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


QualiaComponent = TypeVar("QualiaComponent", bound=AbstractQualiaComponent)
