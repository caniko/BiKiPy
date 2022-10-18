import datetime
from logging import getLogger
from math import floor

from bikipy.behaviour.core.base import BaseTrial
from bikipy.behaviour.mixin.live import LiveTrial
from bikipy.perimeter.base import SinglePerimeter

logger = getLogger(__name__)


class InfinityMaze(BaseTrial, LiveTrial):
    regression_buffer = datetime.time(microsecond=100000)

    def __init__(
        self,
        choice: SinglePerimeter,
        reward_left: SinglePerimeter,
        reward_right: SinglePerimeter,
        return_left: SinglePerimeter,
        return_right: SinglePerimeter,
        delay_entry: SinglePerimeter,
        delay_zone: SinglePerimeter,
        regression_seconds_tolerance: float = 1.5,
        regression_instance_tolerance: int = 3,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.choice = choice
        self.reward_left = reward_left
        self.reward_right = reward_right
        self.return_left = return_left
        self.return_right = return_right
        self.delay_entry = delay_entry
        self.delay_zone = delay_zone

        self.perimeters = {
            "choice": self.choice,
            "reward_left": self.reward_left,
            "reward_right": self.reward_right,
            "return_left": self.return_left,
            "return_right": self.return_right,
            "delay_entry": self.delay_entry,
            "delay_zone": self.delay_zone,
        }

        regression_seconds_tolerance_decimal = regression_seconds_tolerance % 1.0
        self._regression_seconds_tolerance = datetime.time(
            second=floor(regression_seconds_tolerance),
            microsecond=round(regression_seconds_tolerance_decimal * 10**6),
        )

        self.regression_data = []
        self.quasi_regression_data = []

        self.reward_data = []

        self.regression_instance_tolerance = int(regression_instance_tolerance)
        self.regressed = False
        self._sequential_regressions = 0
        self._regressing = False
        self._regression_start_timestamp = None

        self._last_location_id = None
        self._last_node = None
        self._current_sequence = []

        self._current_loop_is_bad = False

        self._last_loop = None
        self._received_reward = False

    @property
    def state_string(self):
        return super().state_string + f"; Regressed: {self.regressed}"

    def localize_loop_func(self, location: int, timestamp: datetime.datetime):
        if location == self._last_location_id:
            if self._regressing and self._regression_start_timestamp >= self.regression_buffer:
                self._regressing = False
                self.regressed = True
            return

        if self._regressing:
            self.quasi_regression_data.append(
                (
                    self._regression_start_timestamp,
                    timestamp - self._regression_start_timestamp,
                )
            )
            self._regressing = False
            self._regression_start_timestamp = None
        elif self.regressed:
            self.regression_data.append(
                (
                    self._regression_start_timestamp,
                    timestamp - self._regression_start_timestamp,
                )
            )
            self.regressed = False
            self._sequential_regressions += 1

        location_string = self._int_id_to_perimeter_label[location] if location else None
        if location_string is None:
            pass
        elif "delay_zone" == location_string:
            self.initiate_countdown(self._loop_number_to_delay_time[self.loop_number])
        elif "delay_entry" == location_string:
            if self._last_node == "delay_zone":
                self.stop_countdown_prematurely()
        elif location in self._current_sequence:
            self._regressing = True
            self._regression_start_timestamp = timestamp
        elif "reward" in location_string and not self._current_loop_is_bad:
            if self._received_reward:
                self.record_bad_loop("The animal regressed to the other reward site after getting a reward")
            elif self._last_loop:
                if "reward_left" == location_string:
                    if "right" == self._last_loop:
                        self.reward("left")
                    else:
                        self.record_bad_loop(f"Made left turn {self.bad_turn_counter} after the initial left turn")

                else:  # same as `elif "reward_right" == location_string:`
                    if "left" == self._last_loop:
                        self.reward("right")
                    else:
                        self.record_bad_loop(f"Made right turn {self.bad_turn_counter} after the initial right turn")
                    self._last_loop = "right"
            else:
                self._last_loop = location_string.split("_")[1]
                logger.info(f"First reward is being delivered on the {self._last_loop}")
                self.reward(self._last_loop)

        self._last_location_id = location
        self._last_node = location_string

        self.node_sequence.append(location)
        if location:
            self._current_sequence.append(location)

    async def countdown(self, seconds: int = 10):
        await super().countdown(seconds)
        self._current_sequence = []
        self._current_loop_is_bad = False
        self.loop_number += 1

    def reward(self, direction: str):
        logger.debug(f"Dropping reward on {direction}")
        self.bad_turn_counter = 0
        self._received_reward = True

        # TODO: Arduino connection

        self.reward_data.append((direction, datetime.datetime.now()))
        logger.debug(f"Reward on {direction} was successful")

    def record_bad_loop(self, reason: str):
        logger.debug(reason)
        self._current_loop_is_bad = True
        self.bad_loop_record[self.loop_number] = reason
