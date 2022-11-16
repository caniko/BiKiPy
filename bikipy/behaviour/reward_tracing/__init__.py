from abc import ABC
from functools import cached_property

import numpy as np
import pandas as pd
from pydantic_numpy import NDArrayBool

from bikipy.behaviour.core.abstract import AbstractTrial
from bikipy.behaviour.core.enclosure.base import EnclosedTrial, EnclosedExperiment
from bikipy.feature.motion import Motion, motion_multi_indexer, EMPTY_MOTION
from bikipy.perimeter.base import Perimeter


class RewardTraceTrialMixin(AbstractTrial, ABC):
    start_perimeter: Perimeter
    reward_perimeter: Perimeter

    temporal_tolerance: float = 1.0 / 3.0

    perimeter_labels = {"start_perimeter", "reward_perimeter"}

    @cached_property
    def _start_frame_idx(self) -> int:
        try:
            return np.where(
                self.start_perimeter.compute_confined_coordinate_boolean_index(
                    self.kinematic_coordinates, potential_label=f"{self.label}_start"
                )
            )[0][0]
        except IndexError:
            return np.nan

    @cached_property
    def _reward_arrival_idx(self) -> int:
        try:
            return np.where(self.reward_boolean)[0][0]
        except IndexError:
            return np.nan

    @cached_property
    def either_start_or_reward_undetected(self) -> bool:
        return np.any(np.isnan((self._start_frame_idx, self._reward_arrival_idx)))

    @cached_property
    def reward_boolean(self) -> NDArrayBool:
        # return single_node_tolerance_filter(
        #     self.reward_perimeter.compute_confined_coordinate_boolean_index(self.kinematic_coordinates), self.video.fps
        # )
        return self.reward_perimeter.compute_confined_coordinate_boolean_index(
            self.kinematic_coordinates, potential_label=f"{self.label}_reward"
        )

    @property
    def start_to_reward_motion(self) -> tuple:
        return (
            EMPTY_MOTION
            if self.either_start_or_reward_undetected
            else Motion(
                coordinate_sequence=self.kinematic_coordinates[self._start_frame_idx : self._reward_arrival_idx],
                fps=self.video.fps,
            ).as_tuple
        )

    @property
    def seconds_to_find_reward(self) -> float:
        """
        Seconds taken to find reward
        :return:
        """
        if self.either_start_or_reward_undetected:
            return np.nan
        return (self._reward_arrival_idx - self._start_frame_idx) / self.video.fps

    @property
    def seconds_spent_in_reward_area(self) -> float:
        return np.sum(self.reward_boolean) / self.video.fps

    @property
    def _trial_feature_series_list(self) -> list[pd.Series]:
        upstream_list = super()._trial_feature_series_list

        category = "Cheeseboard"
        upstream_list.append(
            pd.Series(
                [self.seconds_spent_in_reward_area, self.seconds_to_find_reward, *self.start_to_reward_motion],
                index=[
                    (category, "RewardAreaSeconds"),
                    (category, "RewardTraceSeconds"),
                    *motion_multi_indexer("StartToReward", 2),
                ],
            )
        )

        return upstream_list


class GenericRewardTraceTrial(EnclosedTrial, RewardTraceTrialMixin):
    trial_label = "reward_trace"


class GenericRewardTraceExperiment(EnclosedExperiment):
    experiment_labels = {"reward", "reward_trace"}

    trial_sequence = (GenericRewardTraceTrial,)
