import asyncio
import logging
from collections import Sequence as collections_Sequence
from functools import cached_property
from logging import getLogger
from typing import Sequence, Union

import numpy as np

from bikipy.behaviour.live import LiveTrial
from bikipy.perimeter.base import Perimeter2D
from bikipy.utils.store import RangeDict

logger = getLogger(__name__)


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
        delay_timings_trial_count: Union[int, Sequence[int]],
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

        self.last_location = None
        self.node_sequence = []
        self.bad_loop_record = {}

        self._loop_number = 0
        self._loop_number_vs_delay_time = RangeDict({
            i: delay_time for i, delay_time in
            zip(self.delay_timings_trial_count, self.delay_timings)
        })
        self._bad_turn_counter = 0
        self._last_loop = None
        self._received_reward = False
        self._delay_countdown_task = None

    async def countdown(self, seconds: int = 10):
        await super().countdown(seconds)
        self._loop_number += 1

    async def localize_loop_func(self, location: int):
        if location == self.last_location or not location:
            return

        location_string = self._int_id_vs_perimeter_label[location]

        if "reward" in location_string:
            if self._last_loop:
                if "reward_left" == self.last_location:
                    if "right" == self._last_loop:
                        self.reward()
                    else:
                        self.record_bad_loop(
                            f"Made left turn {self._bad_turn_counter} after the initial left turn"
                        )
                else:
                    if "left" == self._last_loop:
                        self.reward()
                    else:
                        self.record_bad_loop(
                            f"Made right turn {self._bad_turn_counter} after the initial right turn"
                        )
            else:
                self.reward()

        if self._delay_countdown_task is None or self._delay_countdown_task.done():
            if location == self._perimeter_label_vs_int_id["delay"]:
                self._delay_countdown_task = asyncio.create_task(
                    self.countdown(self._loop_number_vs_delay_time[self._loop_number])
                )
        elif location == self._perimeter_label_vs_int_id["entry"] and not self._delay_countdown_task.done():
            self._delay_countdown_task.cancel()

        self.node_sequence.append(location)
        self.last_location = location

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
