from functools import cached_property

import numpy as np
import pandas as pd
from pydantic_numpy import NDArrayBool

from bikipy.behaviour.core.enclosure.circle import (
    CircleEnclosedTrial,
    CircleEnclosedExperiment,
    CircleEnclosedHabituationTrial,
)
from bikipy.feature.tolerance.single import single_node_tolerance_filter
from bikipy.perimeter.base import Perimeter
from bikipy.perimeter.radial.circle import CirclePerimeter


class CheeseboardTrial(CircleEnclosedTrial):
    start_perimeter: Perimeter
    reward_perimeter: CirclePerimeter

    temporal_tolerance: float = 1.0 / 3.0

    @cached_property
    def reward_boolean(self) -> NDArrayBool:
        return single_node_tolerance_filter(
            self.reward_perimeter.compute_confined_coordinate_boolean_index(self.kinematic_coordinates), self.video.fps
        )

    @property
    def seconds_to_find_reward(self) -> float:
        """
        Seconds taken to find reward
        :return:
        """
        start_boolean = single_node_tolerance_filter(
            self.start_perimeter.compute_confined_coordinate_boolean_index(self.kinematic_coordinates), self.video.fps
        )

        trace_start_idx = np.where(start_boolean)[0]
        reward_arrival_idx = np.where(self.reward_boolean)[0]

        return (reward_arrival_idx - trace_start_idx) * self.video.fps

    @property
    def seconds_spent_in_reward_area(self) -> float:
        return np.sum(self.reward_boolean) / self.video.fps

    @property
    def _trial_feature_series_list(self) -> list[pd.Series]:
        upstream_list = super()._trial_feature_series_list

        upstream_list.append(
            pd.Series(
                [self.seconds_to_find_reward, self.seconds_spent_in_reward_area],
                index=[("Cheeseboard", "RewardTraceSeconds"), ("Cheeseboard", "RewardAreaSeconds")],
            )
        )

        return upstream_list


class CheeseboardExperiment(CircleEnclosedExperiment):
    habituation_trial_class = CircleEnclosedHabituationTrial

    trial_classes = (CheeseboardTrial,)
    trial_sequence_repetition = ...
