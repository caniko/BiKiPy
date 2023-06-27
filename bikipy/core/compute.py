from abc import ABC, abstractmethod
from pathlib import Path
from typing import ClassVar, Generic, Optional, Type, TypeVar

from matplotlib.axes import Axes
from pydantic import Extra, root_validator, validate_arguments
from pydantic.generics import GenericModel
from pydantic_numpy import NDArrayBool

from bikipy.core.base import BikipyModel
from bikipy.core.video import VideoMetadata, VideoMetadataMixin
from bikipy.utils.plot import BOTTOM_LEGEND_KWARGS

T = TypeVar("T")


class AbstractCompute(GenericModel, Generic[T], BikipyModel, ABC):
    label: str = ...

    heuristic_data_sources: ClassVar[tuple[str, ...]]

    class Config:
        extra = Extra.allow

    @root_validator
    def check_at_least_one_field(cls, values):
        if not cls.heuristic_data_sources:
            msg = f"Ask project authors to define heuristic_data_sources for the {cls.__name__} class"
            raise AttributeError(msg)
        if not any(field in values for field in cls.heuristic_data_sources):
            msg = f"One of {cls.heuristic_data_sources} must be defined"
            raise AttributeError(msg)

        return values

    @property
    @abstractmethod
    def result(self) -> T:
        ...

    @abstractmethod
    def plot(self, ax: Axes, *args, **kwargs) -> None:
        ...

    def plot_finalization(self, ax: Axes, video: Optional[VideoMetadata] = None) -> None:
        ax.set_title(self.label)
        ax.legend(**BOTTOM_LEGEND_KWARGS)

    @validate_arguments
    def save_fig(self, path: Path, video: VideoMetadata, **plot_kwargs) -> None:
        fig, ax = video.subplot()
        self.plot(ax, **plot_kwargs)
        fig.tight_layout()
        fig.savefig(path)


class AbstractComputeBooleanIndex(AbstractCompute[NDArrayBool], VideoMetadataMixin, ABC):
    tolerance_modelling: bool = True

    @property
    def result_seconds(self) -> float:
        return self.video.boolean_array_to_seconds(self.result)


ComputeBooleanIndexCLS = Type[AbstractComputeBooleanIndex]
ComputeBooleanIndex = TypeVar("ComputeBooleanIndex", bound=AbstractComputeBooleanIndex)
