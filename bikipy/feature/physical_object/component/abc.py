from abc import ABC, abstractmethod
from functools import cached_property
from logging import getLogger
from typing import Any, ClassVar, TypeVar, Type

import pandas as pd
from pydantic import Field
from pydantic_numpy.dtype import NDArrayBool

from bikipy.core.base_class import BaseBikipyInspectMixin
from bikipy.core.video import VideoMetadata, VideoMetadataMixin
from bikipy.feature.attention.model import AttentionModelMixin
from bikipy.feature.tolerance.single import single_node_tolerance_model
from bikipy.perimeter.base import SinglePerimeter
from bikipy.reader.base import Reader

logger = getLogger(__name__)


class AbcQualiaComponent(BaseBikipyInspectMixin, VideoMetadataMixin, AttentionModelMixin, ABC):
    perimeter: SinglePerimeter = ...
    reader: Reader = ...
    filter_in_sequence: bool = Field(
        False,
        description="When set to True, the component boolean index will be "
        "considered in sequence with other components that are also filtered in sequence",
    )

    axes_row: Any = ...

    native_inspection_row_length: ClassVar[int] = ...
    component_label: ClassVar[str] = ...

    @classmethod
    @property
    def inspection_row_length(cls) -> int:
        return cls.native_inspection_row_length + 2

    def _combined_sensation_plot(self, result: NDArrayBool) -> None:
        self.generic_result_plotter(result, self.axes_row[-2], "Combined")

    @property
    @abstractmethod
    def boolean_index(self) -> NDArrayBool:
        ...

    @property
    @abstractmethod
    def summary_series(self) -> pd.Series:
        ...

    @cached_property
    def seconds_of_observation_qualia(self) -> float:
        return self.boolean_array_to_seconds(self.boolean_index)

    @cached_property
    def tolerance_modeled_combined_sensation(self) -> NDArrayBool:
        result = single_node_tolerance_model(self.boolean_index, self.video.fps)
        self.generic_result_plotter(result, self.axes_row[-1], "ToleranceModeledCombined")
        return result

    @cached_property
    def tolerance_modeled_combined_sensation_seconds(self) -> float:
        return self.boolean_array_to_seconds(self.tolerance_modeled_combined_sensation)

    @cached_property
    def tolerance_vs_unfiltered_ratio(self) -> float:
        return self.tolerance_modeled_combined_sensation_seconds / self.seconds_of_observation_qualia

    def _summary_indexer(self, data_labels: list[str], with_component_label: bool = True) -> pd.MultiIndex:
        if with_component_label:
            additive = self.component_label.capitalize()
            data_labels = [f"{additive}{label}" for label in data_labels]
        return pd.MultiIndex.from_product([[self.perimeter.label], data_labels])

    @property
    def component_summary(self) -> pd.Series:
        base = pd.Series(
            [
                self.seconds_of_observation_qualia,
                self.tolerance_modeled_combined_sensation_seconds,
                self.tolerance_vs_unfiltered_ratio,
            ],
            index=self._summary_indexer(["CombinedSeconds", "TolCombinedSeconds", "TolUnfilteredRatio"]),
        )
        return pd.concat([pd.Series(self.summary_series), base])

    @cached_property
    def video(self) -> VideoMetadata:
        return VideoMetadata.join(self.perimeter.video, self.reader.video, ignore_incongruity=True)

    def generic_result_plotter(self, valid_boolean_index: NDArrayBool, ax: Any, label: str) -> None:
        ax.set_title(label, fontsize=self.video.upscaled_video.plotting_title_font_size)
        ax.scatter(*self.reader.kinematic_coordinates[valid_boolean_index].T, marker="x", color="green")

    def __len__(self) -> int:
        return self.reader.frames


QualiaComponentType = Type[AbcQualiaComponent]
QualiaComponent = TypeVar("QualiaComponent", bound=AbcQualiaComponent)
