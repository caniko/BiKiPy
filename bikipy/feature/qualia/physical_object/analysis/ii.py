from functools import cached_property
from itertools import permutations
from logging import getLogger
from typing import ClassVar

import numpy as np
import pandas as pd
from pydantic import computed_field
from pydantic_numpy.typing import NpNDArrayUint8

from bikipy.feature.qualia.physical_object.analysis.i import (
    OnePhysicalObjectSetQualiaAnalysis,
)
from bikipy.math.discrete import reduce_repeating_sequences

logger = getLogger(__name__)


class TwoPhysicalObjectSetQualiaAnalysis(OnePhysicalObjectSetQualiaAnalysis):
    overlapping_frame_to_total_frame_warning_ratio: ClassVar[float] = 0.05

    @computed_field  # type: ignore[misc]
    @property
    def _analysis_series_list(self) -> list[pd.Series]:
        result = super()._analysis_series_list
        result.append(
            pd.Series(
                {
                    **{
                        f"AbsoluteDiscrimination{po_label_a.capitalize()}{po_label_b.capitalize()}": discrimination
                        for (
                            po_label_a,
                            po_label_b,
                        ), discrimination in self.pair_to_absolute_object_discrimination.items()
                    },
                    **{
                        f"BiasScore{physical_object_label.capitalize()}": rel_bias_score
                        for physical_object_label, rel_bias_score in self.relative_object_bias_score.items()
                    },
                    **{
                        f"AbsoluteBiasScore{physical_object_label.capitalize()}": abs_bias_score
                        for physical_object_label, abs_bias_score in self.absolute_object_bias_score.items()
                    },
                    "TotalObservationInstances": self.po_sum_of_observation_instances,
                }
            )
        )
        return result

    @computed_field  # type: ignore[misc]
    @cached_property
    def pair_to_absolute_object_discrimination(self) -> dict[tuple[str, str], float]:
        return {
            (perm_a, perm_b): abs(
                self.po_label_to_seconds_observing[perm_a] - self.po_label_to_seconds_observing[perm_b]
            )
            for perm_a, perm_b in permutations(self.po_label_to_seconds_observing, 2)
        }

    @computed_field  # type: ignore[misc]
    @cached_property
    def relative_object_bias_score(self) -> dict[str, float]:
        if not self.po_total_seconds_observing:
            return self._po_label_to_zero
        return {
            physical_object_label: 100.0 * frames_observing / self.frames_observing
            for physical_object_label, frames_observing in self.po_label_to_frames_observing.items()
        }

    @computed_field  # type: ignore[misc]
    @property
    def object_bias_score(self) -> dict[str, float]:
        """
        Academic literature, and statistics concludes relative_object_bias_score is the best for comparison across
        animals.
        """
        return self.relative_object_bias_score

    @computed_field  # type: ignore[misc]
    @cached_property
    def absolute_object_bias_score(self) -> dict[str, float]:
        if not self.po_total_seconds_observing:
            return self._po_label_to_zero
        return {
            physical_object_label: 100.0 * frames_observing / self.frames
            for physical_object_label, frames_observing in self.po_label_to_frames_observing.items()
        }

    @computed_field  # type: ignore[misc]
    @cached_property
    def po_observation_sequence(self) -> NpNDArrayUint8:
        overlapping_frames = 0

        observation_sequence = np.zeros(self.frames, dtype=np.uint8)
        for object_int_id, object_boolean_index in enumerate(self.po_label_to_qualia_boolean_index.values(), start=1):
            overlapping_frames += np.sum(object_boolean_index & observation_sequence)
            observation_sequence[object_boolean_index] = object_int_id  # TODO: Revert

        if overlapping_frames:
            ratio = overlapping_frames / self.frames
            logger.info(f"{overlapping_frames} out of {self.frames} overlap; ratio {overlapping_frames / self.frames}")
            if ratio > self.overlapping_frame_to_total_frame_warning_ratio:
                logger.warning(
                    f"Ratio is above the warning ratio, {self.overlapping_frame_to_total_frame_warning_ratio}:\n"
                    "This is a soft warning; do not be alarmed. Two or more objects have a temporal "
                    "overlap in their observation_sequence. This warning can be  ignored in most instances. "
                    "You have been warned that this observation_sequence data "
                    "MIGHT be empirically wrong. You may re-annotate the perimeters "
                    "of the objects for improved results"
                )

        return observation_sequence

    @computed_field  # type: ignore[misc]
    @cached_property
    def po_sum_of_observation_instances(self) -> int:
        return sum(
            e != 0
            for e in reduce_repeating_sequences(
                self.po_observation_sequence,
                minimum_repeating=self.video.minimum_frames_tolerance,
            )
        )
