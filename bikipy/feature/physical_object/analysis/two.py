from functools import cached_property

import numpy as np

from bikipy.feature.physical_object.analysis.one import (
    OnePhysicalObjectSetQualiaAnalysis,
)


class TwoPhysicalObjectSetQualiaAnalysis(OnePhysicalObjectSetQualiaAnalysis):
    @cached_property
    def absolute_object_discrimination(self) -> float:
        seconds = tuple(self.physical_object_label_to_observation_seconds.values())
        return abs(seconds[0] - seconds[1])

    @property
    def object_bias_score(self) -> dict[str, float]:
        """
        Academic literature, and statistics concludes relative_object_bias_score is the best for comparison across
        animals.
        """
        return self.relative_object_bias_score

    @cached_property
    def absolute_object_bias_score(self) -> dict[str, float]:
        if not self.total_seconds_observing:
            return self._physical_object_label_to_zero
        return {
            label: 100.0 * observation_boolean_index / (self.frames * self.video.fps)
            for label, observation_boolean_index in self.physical_object_label_to_observation_boolean_index.items()
        }

    @cached_property
    def relative_object_bias_score(self) -> dict[str, float]:
        if not self.total_seconds_observing:
            return self._physical_object_label_to_zero
        return {
            label: 100.0 * np.sum(observation_boolean_index) / self.total_seconds_observing
            for label, observation_boolean_index in self.physical_object_label_to_observation_boolean_index.items()
        }
