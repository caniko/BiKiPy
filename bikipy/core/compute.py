from abc import ABC, abstractmethod
from typing import ClassVar

import numpy as np
from matplotlib.axes import Axes
from pydantic import Extra, computed_field, model_validator
from pydantic_numpy.typing import Np1DArrayBool

from bikipy.core.base import BikipyModel
from bikipy.core.video import VideoMetadataMixin
from bikipy.perimeter.base import BasePerimeter
from bikipy.plot import BOTTOM_LEGEND_KWARGS


class AbstractCompute[ResultType](BikipyModel, ABC, extra=Extra.allow):
    label: str

    heuristic_data_sources: ClassVar[tuple[str, ...]]
    heuristic_data_sources_all_required: ClassVar[bool] = False

    @model_validator(mode="before")
    def check_at_least_one_field(cls, values):
        if not hasattr(cls, "heuristic_data_sources"):
            msg = f"Ask project authors to define heuristic_data_sources for the {cls.__name__} class"
            raise AttributeError(msg)

        method = all if cls.heuristic_data_sources_all_required else any
        if not method(field in values for field in cls.heuristic_data_sources):
            msg = f"One of {cls.heuristic_data_sources} must be defined"
            raise AttributeError(msg)

        return values

    def plot_finalization(self, ax: Axes) -> None:
        ax.set_title(self.label)
        ax.legend(**BOTTOM_LEGEND_KWARGS)

    @property
    @abstractmethod
    def result(self) -> ResultType: ...

    @abstractmethod
    def plot(self, ax: Axes) -> None: ...


class AbstractComputePerimeterBooleanIndex[P: BasePerimeter](AbstractCompute[Np1DArrayBool], VideoMetadataMixin, ABC):
    perimeter: P
    tolerance_modelling: bool = True

    @computed_field  # type: ignore[misc]
    @property
    def result_seconds(self) -> float:
        return self.video.boolean_array_to_seconds(self.result)

    @property
    @abstractmethod
    def perimeter_to_boolean_index(self) -> dict[P, Np1DArrayBool]: ...

    def video_gen_merge_perimeter_to_boolean_index(
        self, other: "AbstractComputePerimeterBooleanIndex", both_or_false: bool = False
    ) -> dict[P, Np1DArrayBool]:
        return video_gen_merge_perimeter_to_boolean_index_from_dict(
            self.perimeter_to_boolean_index, other.perimeter_to_boolean_index, both_or_false
        )


def video_gen_merge_perimeter_to_boolean_index_from_dict[
    P: BasePerimeter
](
    a_perimeter_to_boolean_index,
    b_perimeter_to_boolean_index: dict[P, Np1DArrayBool],
    both_or_false: bool = False,
) -> dict[P, Np1DArrayBool]:
    result = {**a_perimeter_to_boolean_index, **b_perimeter_to_boolean_index}

    logical_method = np.logical_and if both_or_false else np.logical_or
    for common_key in frozenset(a_perimeter_to_boolean_index).intersection(b_perimeter_to_boolean_index):
        result[common_key] = logical_method(
            a_perimeter_to_boolean_index[common_key], b_perimeter_to_boolean_index[common_key]
        )

    return result
