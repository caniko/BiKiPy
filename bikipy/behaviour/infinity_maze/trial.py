import asyncio
from collections import Sequence as collections_Sequence
import datetime
from functools import cached_property
from logging import getLogger
from math import floor
from typing import Sequence, Union

import numpy as np

from bikipy.behaviour.live import LiveTrial
from bikipy.perimeter.base import Perimeter2D
from bikipy.utils.store import RangeDict

logger = getLogger(__name__)


class InfinityMaze(LiveTrial):
    regression_buffer = datetime.time(microsecond=100000)

    def __init__(
        self,
        delay_perimeter: Perimeter2D,
        choice_perimeter: Perimeter2D,
        reward_left_perimeter: Perimeter2D,
        reward_right_perimeter: Perimeter2D,
        return_left_perimeter: Perimeter2D,
        return_right_perimeter: Perimeter2D,
        delay_perimeter_perimeter: Perimeter2D,
        delay_timings: Sequence[Union[float, int]],
        delay_timings_trial_count: Union[int, Sequence[int]],
        regression_seconds_tolerance: Union[float, int] = 1.5,
        regression_instance_tolerance: int = 3,
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
        self.delay_entry_perimeter = delay_perimeter_perimeter

        self.perimeters = {
            "delay": self.delay_perimeter,
            "choice": self.choice_perimeter,
            "reward_left": self.reward_left_perimeter,
            "reward_right": self.reward_right_perimeter,
            "return_left": self.return_left_perimeter,
            "return_right": self.return_right_perimeter,
            "delay_entry": self.delay_entry_perimeter,
        }

        assert all(isinstance(obj, Perimeter2D) for obj in self.perimeters.values())

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

        self.node_sequence = []
        self.bad_loop_record = {}

        self._loop_number_vs_delay_time = RangeDict(
            {
                i: delay_time
                for i, delay_time in zip(
                    self.delay_timings_trial_count, self.delay_timings
                )
            }
        )

        regression_seconds_tolerance_decimal = regression_seconds_tolerance % 1.0
        self._regression_seconds_tolerance = datetime.time(
            second=floor(regression_seconds_tolerance),
            microsecond=regression_seconds_tolerance_decimal * 10**6
        )

        self.regression_data = []
        self.quasi_regression_data = []

        self.regression_instance_tolerance = int(regression_instance_tolerance)
        self._sequential_regressions = 0
        self._regressing = False
        self._regressed = False
        self._regression_start_timestamp = None

        self._last_location = None
        self._last_node = None
        self._current_sequence = []

        self._loop_number = 0
        self._bad_turn_counter = 0
        self._last_loop = None
        self._received_reward = False
        self._delay_countdown_task = None

    async def countdown(self, seconds: int = 10):
        await super().countdown(seconds)
        self._current_sequence = []
        self._loop_number += 1

    async def localize_loop_func(self, location: int):
        if location == self._last_location:
            if self._regressing and self._regression_start_timestamp >= self.regression_buffer:
                self._regressing = False
                self._regressed = True
            return

        if self._regressing:
            self.quasi_regression_data.append((
                self._regression_start_timestamp,
                datetime.datetime.now() - self._regression_start_timestamp
            ))
            self._regressing = False
            self._regression_start_timestamp = None
        elif self._regressed:
            self.regression_data.append((
                self._regression_start_timestamp,
                datetime.datetime.now() - self._regression_start_timestamp
            ))
            self._regressed = False
            self._sequential_regressions += 1

        if "delay" == (location_string := self._int_id_vs_perimeter_label[location]):
            self._delay_countdown_task = asyncio.create_task(
                self.countdown(self._loop_number_vs_delay_time[self._loop_number])
            )
        elif "delay_entry" == location_string:
            if self._last_node == "delay":
                self._delay_countdown_task.cancel()
        elif location in self._current_sequence:
            self._regressing = True
            self._regression_start_timestamp = datetime.datetime.now()
        elif "reward" in location_string:
            if self._last_loop:
                if "reward_left" == location_string:
                    if "right" == self._last_loop:
                        self.reward()
                    else:
                        self.record_bad_loop(
                            f"Made left turn {self._bad_turn_counter} after the initial left turn"
                        )
                else:  # same as `elif "reward_right" == location_string:`
                    if "left" == self._last_loop:
                        self.reward()
                    else:
                        self.record_bad_loop(
                            f"Made right turn {self._bad_turn_counter} after the initial right turn"
                        )
            else:
                self.reward()

        self._last_location = location_string
        self._last_node = location_string

        self.node_sequence.append(location)
        if location:
            self._current_sequence.append(location)

    def reward(self):
        self._bad_turn_counter = 0
        # TODO: Arduino connection
        return

    def record_bad_loop(self, reason: str):
        logger.debug(reason)
        self.bad_loop_record[self._loop_number] = reason

    @cached_property
    def _left_loop_int_ids(self):
        return self._perimeter_label_sequence_to_int_id(self.left_loop)

    @cached_property
    def _right_loop_int_ids(self):
        return self._perimeter_label_sequence_to_int_id(self.right_loop)
