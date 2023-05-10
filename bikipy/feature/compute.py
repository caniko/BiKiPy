from abc import ABC, abstractmethod
from pathlib import Path
from typing import Generic, Type, TypeVar

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from pydantic import validate_arguments, Extra
from pydantic.generics import GenericModel
from pydantic_numpy import NDArrayBool

from bikipy.core.base import BikipyModel
from bikipy.core.video import VideoMetadata, VideoMetadataMixin

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

    @validate_arguments
    def save_fig(self, path: Path, video: VideoMetadata, **plot_kwargs) -> None:
        fig, ax = video.subplot()
        self.plot(ax, **plot_kwargs)
        plt.tight_layout()
        plt.savefig(path)


class AbstractComputeBooleanIndex(AbstractCompute[NDArrayBool], VideoMetadataMixin, ABC):
    tolerance_modelling: bool = True

    @property
    def result_seconds(self) -> float:
        return self.boolean_array_to_seconds(self.result)


ComputeBooleanIndexCLS = Type[AbstractComputeBooleanIndex]
ComputeBooleanIndex = TypeVar("ComputeBooleanIndex", bound=AbstractComputeBooleanIndex)
