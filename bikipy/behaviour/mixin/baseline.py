from abc import abstractmethod, ABC
from functools import cached_property
from typing import ClassVar

import pandas as pd

from bikipy.core.base_class import BikipyBase


class MotionBaselineExperimentMixin(BikipyBase, ABC):
    baseline_stage: str

    @property
    @abstractmethod
    def animal_id_indexed_motion_summary_frame(self):
        ...

    @property
    @abstractmethod
    def stages(self):
        ...

    @cached_property
    def baseline_delta_motion_frame(self):
        self._baseline_assertion()

        data = []
        for stage in self.stages:
            if stage == self.baseline_stage:
                continue
            data.append(
                self._baseline_motion_frame
                - self.animal_id_indexed_motion_summary_frame[stage]
            )
        return pd.concat(data)

    @cached_property
    def _baseline_motion_frame(self):
        return self.animal_id_indexed_motion_summary_frame[self.baseline_stage].copy()

    def _baseline_assertion(self):
        assert self.baseline_stage
        assert self.baseline_stage in self.stages


class FeatureBaselineExperimentMixin(MotionBaselineExperimentMixin, ABC):
    baseline_delta_features: ClassVar[tuple[str]]

    @property
    @abstractmethod
    def animal_id_indexed_experiment_specific_feature_frame(self):
        ...

    @property
    @abstractmethod
    def _feature_frame_columns(self):
        ...

    @cached_property
    def baseline_delta_feature_frame(self) -> pd.DataFrame:
        df = self.baseline_delta_experiment_specific_feature_frame
        df.columns = self._feature_frame_columns(
            levels=self.baseline_delta_motion_frame.columns.nlevels
        )
        return df.join(self.baseline_delta_motion_frame, how="inner")

    @cached_property
    def baseline_delta_experiment_specific_feature_frame(self):
        self._baseline_assertion()

        data = []
        for stage in self.stages:
            if stage == self.baseline_stage:
                continue
            to_subtract = self.animal_id_indexed_experiment_specific_feature_frame[stage]
            if self.baseline_delta_features:
                to_subtract = to_subtract[self.baseline_stage]
            data.append(
                self._baseline_experiment_specific_feature_frame
                - to_subtract
            )
        return pd.concat(data)

    @cached_property
    def _baseline_experiment_specific_feature_frame(self):
        result = self.animal_id_indexed_experiment_specific_feature_frame[self.baseline_stage]
        if self.baseline_delta_features:
            result = result[self.baseline_stage]
        return result
