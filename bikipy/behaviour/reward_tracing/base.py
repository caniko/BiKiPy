from abc import ABC
from functools import cached_property
from logging import getLogger
from typing import ClassVar, Union

import numpy as np
import pandas as pd
from pydantic import computed_field
from pydantic_numpy.typing import Np1DArrayBool

from bikipy.behaviour.core.constant import ExperimentStage
from bikipy.behaviour.core.enclosure.base import EnclosedExperiment, EnclosedTrial
from bikipy.core.base import BikipyModel
from bikipy.feature.motion import EMPTY_MOTION, Motion, motion_analysis_indexer
from bikipy.feature.tolerance.single import single_node_tolerance_model
from bikipy.perimeter.base import BaseSinglePerimeter

logger = getLogger(__name__)


class RewardTraceTrialMixin[StartPerimeter: BaseSinglePerimeter, RewardPerimeter: BaseSinglePerimeter](
    BikipyModel, ABC
):
    start_perimeter: StartPerimeter
    reward_perimeter: RewardPerimeter

    tolerate_boolean_index: bool = True

    perimeter_labels: ClassVar[str] = {"start_perimeter", "reward_perimeter"}

    @computed_field  # type: ignore[misc]
    @cached_property
    def _start_frame_idx(self) -> Union[int, np.nan]:
        confinement_bool = self.start_perimeter.confinement_boolean_index(
            "reward-trace-start",
            self.reader.kinematic_coordinates,
        )
        if not np.any(confinement_bool):
            return np.nan

        if self.tolerate_boolean_index:
            confinement_bool = single_node_tolerance_model(confinement_bool, self.fps)

        was_confinement = False
        for i, b in enumerate(confinement_bool):
            if b:
                was_confinement = True
            elif was_confinement:  # and not b
                logger.debug(f"Subject {self.label} was in the start area, and then left the starting area")
                return i

        logger.debug(f"Subject {self.label} never left, or was never was in start area")
        return np.nan

    @computed_field  # type: ignore[misc]
    @cached_property
    def _reward_arrival_idx(self) -> int:
        """
        Iterate through boolean index till value is greater than start
        :return:
        """
        for idx in np.where(self.reward_boolean)[0]:
            if idx > self._start_frame_idx:
                return idx
        return np.nan

    @computed_field  # type: ignore[misc]
    @cached_property
    def either_start_or_reward_undetected(self) -> bool:
        return np.any(np.isnan((self._start_frame_idx, self._reward_arrival_idx)))

    @computed_field  # type: ignore[misc]
    @cached_property
    def reward_boolean(self) -> Np1DArrayBool:
        confinement_bool = self.reward_perimeter.confinement_boolean_index(
            "reward-trace-target",
            self.reader.kinematic_coordinates,
        )
        if self.tolerate_boolean_index:
            confinement_bool = single_node_tolerance_model(confinement_bool, self.fps)

        return confinement_bool

    @computed_field  # type: ignore[misc]
    @property
    def start_to_reward_motion(self) -> tuple:
        return (
            EMPTY_MOTION
            if self.either_start_or_reward_undetected
            else Motion(
                coordinate_sequence=self.reader.kinematic_coordinates[self._start_frame_idx : self._reward_arrival_idx],
                fps=self.fps,
            ).as_tuple
        )

    @computed_field  # type: ignore[misc]
    @property
    def seconds_to_find_reward(self) -> float:
        """
        Seconds taken to find reward
        :return:
        """
        if self.either_start_or_reward_undetected:
            return np.nan
        return (self._reward_arrival_idx - self._start_frame_idx) / self.fps

    @computed_field  # type: ignore[misc]
    @property
    def seconds_spent_in_reward_area(self) -> float:
        return np.sum(self.reward_boolean) / self.fps

    @computed_field  # type: ignore[misc]
    @property
    def _analysis_series_list(self) -> list[pd.Series]:
        upstream_list = super()._analysis_series_list

        category: ClassVar[str] = "Cheeseboard"
        upstream_list.append(
            pd.Series(
                [self.seconds_spent_in_reward_area, self.seconds_to_find_reward, *self.start_to_reward_motion],
                index=[
                    (category, "RewardAreaSeconds"),
                    (category, "RewardTraceSeconds"),
                    *motion_analysis_indexer("StartToReward", 2),
                ],
            )
        )

        return upstream_list


class GenericRewardTraceTrial(EnclosedTrial, RewardTraceTrialMixin):
    experiment_stage = ExperimentStage.SINGLE


class GenericRewardTraceExperiment(EnclosedExperiment):
    experiment_labels = {"reward", "reward_trace"}

    trial_sequence = (GenericRewardTraceTrial,)
