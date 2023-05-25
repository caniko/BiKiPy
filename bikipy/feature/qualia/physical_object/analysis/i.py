from functools import cached_property
from logging import getLogger
from typing import Type, TypeVar

import numpy as np
import pandas as pd
from pydantic_numpy.dtype import NDArrayBool

from bikipy.core.mixin import AbstractFeatureCollectorMixin
from bikipy.core.video import VideoMetadataMixin

logger = getLogger(__name__)


class OnePhysicalObjectSetQualiaAnalysis(AbstractFeatureCollectorMixin, VideoMetadataMixin):
    po_label_to_qualia_boolean_index: dict[str, NDArrayBool]

    @property
    def _analysis_series_list(self) -> list[pd.Series]:
        return [
            pd.Series(
                {
                    "TotalSecondsObserving": self.po_total_seconds_observing,
                    **{
                        f"SecondsObserving{po_label.capitalize()}": seconds_observing
                        for po_label, seconds_observing in self.po_label_to_seconds_observing.items()
                    },
                }
            )
        ]

    @cached_property
    def _po_label_to_zero(self) -> dict:
        return {po_label: 0.0 for po_label in self.po_label_to_qualia_boolean_index}

    @cached_property
    def frames(self) -> int:
        return len(tuple(self.po_label_to_qualia_boolean_index.values())[0])

    @cached_property
    def po_observing_per_frame(self) -> np.ndarray[bool, bool]:
        return np.logical_or.reduce(
            [observation_boolean_index for observation_boolean_index in self.po_label_to_qualia_boolean_index.values()]
        )

    @cached_property
    def frames_observing(self) -> int:
        return int(np.sum(self.po_observing_per_frame))

    @cached_property
    def po_total_seconds_observing(self) -> float:
        return self.frames_observing / self.video.fps

    @cached_property
    def po_label_to_frames_observing(self) -> dict[str, int]:
        return {
            po_label: np.sum(observation_boolean_index)
            for po_label, observation_boolean_index in self.po_label_to_qualia_boolean_index.items()
        }

    @cached_property
    def po_label_to_seconds_observing(self) -> dict[str, int]:
        return {
            po_label: frames_observing / self.video.fps
            for po_label, frames_observing in self.po_label_to_frames_observing.items()
        }


QualiaAnalysisType = Type[OnePhysicalObjectSetQualiaAnalysis]
QualiaAnalysis = TypeVar("QualiaAnalysis", bound=OnePhysicalObjectSetQualiaAnalysis)
