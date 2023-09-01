from abc import ABC, abstractmethod
from pathlib import Path
from typing import ClassVar, Generic, Optional, TypeVar

import numpy as np
from matplotlib.axes import Axes
from pydantic import Extra, computed_field, root_validator, validate_arguments
from pydantic.generics import GenericModel
from pydantic_numpy import NDArrayBool

from bikipy.core.base import BikipyModel
from bikipy.core.video import VideoMetadata, VideoMetadataMixin
from bikipy.perimeter.base import Perimeter, SinglePerimeter
from bikipy.utils.plot import BOTTOM_LEGEND_KWARGS

T = TypeVar("T")


class AbstractCompute(GenericModel, Generic[T], BikipyModel, ABC):
    label: str = ...

    heuristic_data_sources: ClassVar[tuple[str, ...]]
    heuristic_data_sources_all_required: ClassVar[bool] = False

    class Config:
        extra = Extra.allow

    @root_validator
    def check_at_least_one_field(cls, values):
        if not hasattr(cls, "heuristic_data_sources"):
            msg = f"Ask project authors to define heuristic_data_sources for the {cls.__name__} class"
            raise AttributeError(msg)

        method = all if cls.heuristic_data_sources_all_required else any
        if not method(field in values for field in cls.heuristic_data_sources):
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


class AbstractComputePerimeterBooleanIndex(AbstractCompute[NDArrayBool], VideoMetadataMixin, ABC):
    perimeter: SinglePerimeter = ...
    tolerance_modelling: bool = True

    @computed_field
    @property
    def result_seconds(self) -> float:
        return self.video.boolean_array_to_seconds(self.result)

    @property
    @abstractmethod
    def perimeter_to_boolean_index(self) -> dict[Perimeter, np.ndarray[bool, bool]]:
        ...

    def video_gen_merge_perimeter_to_boolean_index(
        self, other: "AbstractComputePerimeterBooleanIndex", both_or_false: bool = False
    ) -> dict[Perimeter, np.ndarray[bool, bool]]:
        return video_gen_merge_perimeter_to_boolean_index_from_dict(
            self.perimeter_to_boolean_index, other.perimeter_to_boolean_index, both_or_false
        )


def video_gen_merge_perimeter_to_boolean_index_from_dict(
    a_perimeter_to_boolean_index,
    b_perimeter_to_boolean_index: dict[Perimeter, np.ndarray[bool, bool]],
    both_or_false: bool = False,
) -> dict[Perimeter, np.ndarray[bool, bool]]:
    result = {**a_perimeter_to_boolean_index, **b_perimeter_to_boolean_index}

    logical_method = np.logical_and if both_or_false else np.logical_or
    for common_key in frozenset(a_perimeter_to_boolean_index).intersection(b_perimeter_to_boolean_index):
        result[common_key] = logical_method(
            a_perimeter_to_boolean_index[common_key], b_perimeter_to_boolean_index[common_key]
        )

    return result
