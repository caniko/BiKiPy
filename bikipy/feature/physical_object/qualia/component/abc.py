from abc import ABC, abstractmethod
from functools import cached_property
from logging import getLogger
from typing import Any, ClassVar, TypeVar, Type

import pandas as pd
from matplotlib.axes import Axes
from pydantic import Field
from pydantic_numpy.dtype import NDArrayBool

from bikipy.core.mixin import InspectPlotMixin
from bikipy.core.video import VideoMetadataMixin
from bikipy.feature.attention.mixin import AttentionModelMixin
from bikipy.perimeter.base import SinglePerimeter
from bikipy.reader.base import Reader

logger = getLogger(__name__)


class AbstractQualiaComponent(InspectPlotMixin, VideoMetadataMixin, AttentionModelMixin, ABC):
    perimeter: SinglePerimeter = ...
    reader: Reader = ...
    filter_in_sequence: bool = Field(
        False,
        description="When set to True, the component boolean index will be "
        "considered in sequence with other components that are also filtered in sequence",
    )

    ax: Axes = ...

    native_inspection_row_length: ClassVar[int] = ...
    component_label: ClassVar[str] = ...

    @property
    @abstractmethod
    def boolean_index(self) -> NDArrayBool:
        ...

    @cached_property
    def seconds_of_observation_qualia(self) -> float:
        return self.boolean_array_to_seconds(self.boolean_index)

    def __len__(self) -> int:
        return self.reader.frames


QualiaComponentType = Type[AbstractQualiaComponent]
QualiaComponent = TypeVar("QualiaComponent", bound=AbstractQualiaComponent)
