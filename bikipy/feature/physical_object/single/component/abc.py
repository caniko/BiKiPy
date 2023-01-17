from abc import ABC, abstractmethod
from functools import cached_property
from logging import getLogger
from typing import Any, ClassVar

import numpy as np
from pydantic import validator
from pydantic_numpy.dtype import NDArrayBool

from bikipy.core.base_class import BaseBikipyInspectMixin
from bikipy.core.video import VideoMetadata, VideoMetadataMixin
from bikipy.feature.attention.model import AttentionModelMixin
from bikipy.feature.tolerance.single import single_node_tolerance_model
from bikipy.perimeter.base import SinglePerimeter
from bikipy.reader.base import Reader

logger = getLogger(__name__)


class AbcObservationComponent(BaseBikipyInspectMixin, VideoMetadataMixin, AttentionModelMixin, ABC):
    perimeter: SinglePerimeter = ...
    reader: Reader = ...

    tracking_point_label: str = ...
    axes_row: tuple[Any] = ...

    native_inspection_row_length: ClassVar[int] = ...

    @validator("axes_row")
    def axes_row_has_right_length(cls, value: tuple) -> tuple:
        if length := len(value) != cls.inspection_row_length:
            msg = f"axes_row has length {length}, but {cls.inspection_row_length} was expected"
            raise AttributeError(msg)
        return value

    @classmethod
    @property
    def inspection_row_length(cls) -> int:
        return cls.native_inspection_row_length + 2

    def _combined_sensation_plot(self, result: NDArrayBool) -> None:
        self.generic_result_plotter(result, self.axes_row[-2], "Combined")

    @property
    @abstractmethod
    def combined_sensation(self) -> NDArrayBool:
        # _combined_sensation_plot(result)
        ...

    @cached_property
    def combined_sensation_seconds(self) -> float:
        return np.sum(self.combined_sensation) / self.video.fps

    @cached_property
    def tolerance_modeled_combined_sensation(self) -> NDArrayBool:
        result = single_node_tolerance_model(self.combined_sensation, self.video.fps)
        self.generic_result_plotter(result, self.axes_row[-1], "ToleranceModeledCombined")
        return result

    @cached_property
    def tolerance_modeled_combined_sensation_seconds(self) -> float:
        return np.sum(self.tolerance_modeled_combined_sensation) / self.video.fps

    @cached_property
    def tolerance_vs_unfiltered_ratio(self) -> float:
        return self.tolerance_modeled_combined_sensation_seconds / self.combined_sensation_seconds

    @cached_property
    def video(self) -> VideoMetadata:
        return VideoMetadata.join(self.perimeter.video, self.reader.video, ignore_incongruity=True)

    @property
    def label(self):
        return self.perimeter.label

    def generic_result_plotter(self, valid_boolean_index: NDArrayBool, ax: Any, label: str) -> None:
        ax.set_title(label, fontsize=self.video.upscaled_video.plotting_title_font_size)
        ax.scatter(*self.reader[self.label][valid_boolean_index].T, marker="x", color="green")

    def __len__(self) -> int:
        return self.reader.frames
