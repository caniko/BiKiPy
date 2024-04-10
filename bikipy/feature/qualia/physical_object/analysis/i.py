from functools import cached_property
from logging import getLogger
from typing import Self

import numpy as np
import pandas as pd
from pydantic import computed_field
from pydantic_numpy.typing import Np1DArrayBool

from bikipy.core.mixin import AbstractFeatureCollectorMixin
from bikipy.core.video import VideoMetadataMixin
from bikipy.math.shortcut import np_sum_int

logger = getLogger(__name__)

AND_OR_HEURISTIC_INCONGRUENCE_ERROR_MSG = (
    "The provided heuristic analyzers do not have identical physical object objects"
)


class OnePhysicalObjectSetQualiaAnalysis(AbstractFeatureCollectorMixin, VideoMetadataMixin):
    po_label_to_qualia_boolean_index: dict[str, Np1DArrayBool]

    def __and__(self, other) -> Self:
        assert self.__class__ == other.__class__
        try:
            return self.__class__(
                po_label_to_qualia_boolean_index={
                    physical_object_label: qualia_boolean_index
                    & other.po_label_to_qualia_boolean_index[physical_object_label]
                    for physical_object_label, qualia_boolean_index in self.po_label_to_qualia_boolean_index.items()
                }
            )
        except KeyError:
            raise KeyError(AND_OR_HEURISTIC_INCONGRUENCE_ERROR_MSG)

    def __or__(self, other) -> Self:
        assert self.__class__ == other.__class__
        try:
            return self.__class__(
                po_label_to_qualia_boolean_index={
                    physical_object_label: qualia_boolean_index
                    | other.po_label_to_qualia_boolean_index[physical_object_label]
                    for physical_object_label, qualia_boolean_index in self.po_label_to_qualia_boolean_index.items()
                }
            )
        except KeyError:
            raise KeyError(AND_OR_HEURISTIC_INCONGRUENCE_ERROR_MSG)

    def __invert__(self) -> Self:
        return self.__class__(
            po_label_to_qualia_boolean_index={
                physical_object_label: ~qualia_boolean_index
                for physical_object_label, qualia_boolean_index in self.po_label_to_qualia_boolean_index.items()
            }
        )

    @computed_field  # type: ignore[misc]
    @property
    def _analysis_series_list(self) -> list[pd.Series]:
        return [
            pd.Series(
                {
                    "TotalSecondsObserving": self.po_total_seconds_observing,
                    **{
                        f"SecondsObserving{physical_object_label.capitalize()}": seconds_observing
                        for physical_object_label, seconds_observing in self.po_label_to_seconds_observing.items()
                    },
                }
            )
        ]

    @computed_field  # type: ignore[misc]
    @cached_property
    def _po_label_to_zero(self) -> dict:
        return {physical_object_label: 0.0 for physical_object_label in self.po_label_to_qualia_boolean_index}

    @computed_field  # type: ignore[misc]
    @cached_property
    def frames(self) -> int:
        return len(tuple(self.po_label_to_qualia_boolean_index.values())[0])

    @computed_field  # type: ignore[misc]
    @cached_property
    def po_observing_per_frame(self) -> Np1DArrayBool:
        return np.logical_or.reduce(
            [observation_boolean_index for observation_boolean_index in self.po_label_to_qualia_boolean_index.values()]
        )

    @computed_field  # type: ignore[misc]
    @cached_property
    def frames_observing(self) -> int:
        return np_sum_int(self.po_observing_per_frame)

    @computed_field  # type: ignore[misc]
    @cached_property
    def po_total_seconds_observing(self) -> float:
        return self.frames_observing / self.video_for_computation().fps

    @computed_field  # type: ignore[misc]
    @cached_property
    def po_label_to_frames_observing(self) -> dict[str, int]:
        return {
            physical_object_label: np.sum(observation_boolean_index)
            for physical_object_label, observation_boolean_index in self.po_label_to_qualia_boolean_index.items()
        }

    @computed_field  # type: ignore[misc]
    @cached_property
    def po_label_to_seconds_observing(self) -> dict[str, int]:
        return {
            physical_object_label: frames_observing / self.video_for_computation().fps
            for physical_object_label, frames_observing in self.po_label_to_frames_observing.items()
        }


QualiaAnalysisType = type[OnePhysicalObjectSetQualiaAnalysis]
