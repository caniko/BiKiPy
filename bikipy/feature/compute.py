from abc import ABC, abstractmethod
from pathlib import Path
from typing import Generic, Optional, Type, TypeVar

from matplotlib.axes import Axes
from pydantic import Extra, validate_arguments
from pydantic.generics import GenericModel
from pydantic_numpy import NDArrayBool

from bikipy.core.base import BikipyModel
from bikipy.core.video import VideoMetadata, VideoMetadataMixin
from bikipy.utils.plot import BOTTOM_LEGEND_KWARGS

T = TypeVar("T")


class AbstractCompute(GenericModel, Generic[T], BikipyModel, ABC):
    label: str = ...

    class Config:
        extra = Extra.allow

    @property
    @abstractmethod
    def result(self) -> T:
        ...

    @abstractmethod
    def plot(self, ax: Axes, *args, **kwargs) -> None:
        ...

    def plot_finalization(self, ax: Axes, video: Optional[VideoMetadata] = None) -> None:
        font_size = video.upscaled_video.plotting_title_font_size if video else None
        ax.set_title(self.label, fontsize=font_size)
        ax.legend(**BOTTOM_LEGEND_KWARGS, fontsize=font_size)

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
        return self.boolean_array_to_seconds(self.result)


ComputeBooleanIndexCLS = Type[AbstractComputeBooleanIndex]
ComputeBooleanIndex = TypeVar("ComputeBooleanIndex", bound=AbstractComputeBooleanIndex)
