from collections import Sequence as collections_Sequence
from functools import cached_property
from typing import Sequence, Union

import numpy as np

from bikipy.behaviour.live import LiveTrial
from bikipy.perimeter.base import Perimeter2D


class InfinityMaze(LiveTrial):
    def __init__(
        self,
        delay_perimeter: Perimeter2D,
        choice_perimeter: Perimeter2D,
        reward_left_perimeter: Perimeter2D,
        reward_right_perimeter: Perimeter2D,
        return_left_perimeter: Perimeter2D,
        return_right_perimeter: Perimeter2D,
        delay_timings: Sequence[Union[float, int]],
        delay_timings_trial_count: Union[Union[float, int], Sequence[Union[float, int]]],
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.delay_perimeter = delay_perimeter
        self.choice_perimeter = choice_perimeter
        self.reward_left_perimeter = reward_left_perimeter
        self.reward_right_perimeter = reward_right_perimeter
        self.return_left_perimeter = return_left_perimeter
        self.return_right_perimeter = return_right_perimeter

        self.perimeters = {
            "delay": self.delay_perimeter,
            "choice": self.choice_perimeter,
            "reward_left": self.reward_left_perimeter,
            "reward_right": self.reward_right_perimeter,
            "return_left_perimeter": self.return_left_perimeter,
            "return_right_perimeter": self.return_right_perimeter,
        }

        self.start_point = "delay"
        self.left_loop = np.array(
            (self.start_point, "choice", "reward_left", "return_left")
        )
        self.right_loop = np.array(
            (self.start_point, "choice", "reward_right", "return_right")
        )

        self.delay_timings = delay_timings
        if isinstance(delay_timings_trial_count, collections_Sequence):
            self.delay_timings_trial_count = tuple(delay_timings_trial_count)
            assert len(self.delay_timings_trial_count) == len(self.delay_timings)
            assert all(
                isinstance(timing, (float, int))
                for timing in self.delay_timings_trial_count
            )
        elif isinstance(delay_timings_trial_count, (float, int)):
            self.delay_timings_trial_count = tuple(
                delay_timings_trial_count for _ in range(len(delay_timings))
            )
        else:
            msg = f"delay_timings_trial_count has to be a sequence of numbers or number"
            raise ValueError(msg)

    def _localize_loop_func(location: np.ndarray):
        pass

    @cached_property
    def _left_loop_int_ids(self):
        return self._perimeter_label_sequence_to_int_id(self.left_loop)

    @cached_property
    def _right_loop_int_ids(self):
        return self._perimeter_label_sequence_to_int_id(self.right_loop)
