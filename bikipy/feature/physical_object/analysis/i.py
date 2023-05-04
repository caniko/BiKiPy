from functools import cached_property
from logging import getLogger
from typing import ClassVar, TypeVar

import numpy as np
import pandas as pd
from pydantic_numpy.dtype import NDArrayBool, NDArrayUint8

from bikipy.behaviour.utils import reduce_repeating_sequences
from bikipy.core.base import BikipyHashable
from bikipy.core.mixin import AbstractFeatureCollectorMixin
from bikipy.core.video import VideoMetadataMixin

logger = getLogger(__name__)


class OnePhysicalObjectSetQualiaAnalysis(BikipyHashable, AbstractFeatureCollectorMixin, VideoMetadataMixin):
    physical_object_label_to_observation_boolean_index: dict[str, NDArrayBool]

    overlapping_frame_to_total_frame_warning_ratio: ClassVar[float] = 0.05

    @property
    def _analysis_series_list(self) -> list[pd.Series, ...]:
        return [pd.Series({
            "TotalObservingSeconds": self.po_total_seconds_observing,
            **{f"SecondsObserving{label.capitalize()}": seconds_observing for label, seconds_observing in self.po_label_to_seconds_observing.items()},
        })]

    @cached_property
    def _physical_object_label_to_zero(self) -> dict:
        return {label: 0.0 for label in self.physical_object_label_to_observation_boolean_index}

    @cached_property
    def po_observing_per_frame(self) -> NDArrayBool:
        return np.logical_or.reduce(
            [
                observation_boolean_index
                for observation_boolean_index in self.physical_object_label_to_observation_boolean_index.values()
            ]
        )

    @cached_property
    def po_total_seconds_observing(self) -> float:
        return np.sum(self.po_observing_per_frame) / self.video.fps

    @cached_property
    def po_label_to_seconds_observing(self) -> dict[str, int]:
        return {
            label: np.sum(observation_boolean_index) / self.video.fps
            for label, observation_boolean_index in self.physical_object_label_to_observation_boolean_index.items()
        }


QualiaAnalysis = TypeVar("QualiaAnalysis", bound=OnePhysicalObjectSetQualiaAnalysis)
