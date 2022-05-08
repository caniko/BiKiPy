import asyncio
from abc import ABC
from collections import Sequence as collections_Sequence
from datetime import datetime
from functools import cached_property
from logging import getLogger
from typing import Optional, Union

import numpy as np
from pydantic import BaseModel, DirectoryPath, validator
from tqdm import tqdm

from bikipy.utils.misc import clear_console
from bikipy.utils.ranged_dict import RangeDict

try:
    import zmq
    import zmq.asyncio
except ImportError as e:
    msg = (
        "The live module of the package must be installed"
        "for the use of the LiveTrial based classes. Run the following command:\n"
        "pip install BiKiPy[live]"
    )
    raise ImportError(msg) from e

logger = getLogger(__name__)


class LiveTrial(BaseModel, ABC):
    save_root: DirectoryPath
    delay_timings: tuple[Union[float, int]]
    delay_timings_trial_count: Union[tuple[int], int]
    manual_total_loops_per_trial: Optional[int] = None

    node_sequence: list = []

    bad_turn_counter: int = 0
    bad_loop_record: dict = {}

    loop_number: int = 0

    live_expose_metrics: list = []
    countdown_timings: list = []

    _delay_countdown_task = None
    _counting_down = False

    _zmq_context = None
    _socket = None

    @validator("delay_timings_trial_count", "delay_timings")
    def ensure_delay_timings_trial_count_and_delay_timings_have_eq_len(cls, delay_timings_trial_count, delay_timings):
        if isinstance(delay_timings_trial_count, collections_Sequence):
            assert len(delay_timings_trial_count) == len(delay_timings)
            assert all(isinstance(timing, (float, int)) for timing in delay_timings_trial_count)
            delay_timings_trial_count = tuple(delay_timings_trial_count)
        elif isinstance(delay_timings_trial_count, (float, int)):
            delay_timings_trial_count = tuple(delay_timings_trial_count for _ in range(len(delay_timings)))
        return delay_timings_trial_count, delay_timings

    @validator("delay_timings_trial_count", "manual_total_loops_per_trial")
    def manual_total_loops_per_trial_and_delay_timings_trial_count_are_exclusive(
        cls, delay_timings_trial_count, manual_total_loops_per_trial
    ):
        if (
            manual_total_loops_per_trial
            and delay_timings_trial_count
            or not manual_total_loops_per_trial
            and not delay_timings_trial_count
        ):
            msg = (
                "manual_total_loops_per_trial and delay_timings_trial_count are "
                "defined required, but exclusive of each other"
            )
            raise ValueError(msg)
        return delay_timings_trial_count, manual_total_loops_per_trial

    @cached_property
    def _loop_number_vs_delay_time(self):
        return (
            RangeDict(
                {i: delay_time for i, delay_time in zip(self.delay_timings_trial_count, self.delay_timings)},
                allow_less_than_first_key=0,
            )
            if self.delay_timings
            else {}
        )

    @cached_property
    def total_loops_per_trial(self):
        return self.manual_total_loops_per_trial or np.sum(self.delay_timings_trial_count)

    def generate_zmq_context(self, socket_address: str = "tcp://*:5555"):
        self._zmq_context = zmq.asyncio.Context()
        self._socket = self._zmq_context.socket(zmq.PULL)
        self._socket.bind(str(socket_address))

    def application(self):
        event_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(event_loop)

        event_loop.create_task(self.print_state())

        try:
            event_loop.run_until_complete(self.localization_loop())
        except KeyboardInterrupt:
            logger.info("Localization loop interrupted with keyboard interrupt")

        request_save = input("Would you like to save the results from the experiment? Y/n\n").lower()
        if not request_save or request_save == "y":
            self.save()
        else:
            logger.info("Quitting without saving")

    async def localization_loop(
        self,
        data_separator_delimiter: str = ",",
        data_delimiter: str = " ",
    ):
        self.generate_zmq_context()

        try:
            while True:
                message = await self._socket.recv_string()

                coordinate_str, timestamp_str = message.split(data_separator_delimiter)

                if "None" in coordinate_str:
                    continue

                location = self.detect_confined_perimeter(
                    np.array(coordinate_str.split(data_delimiter), dtype=np.float32)
                )
                logger.debug(f"Coordinates: {coordinate_str}\n" f"Timestamp: {timestamp_str}\n" f"Location: {location}")
                self.localize_loop_func(location, datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S.%f"))
        finally:
            self._socket.close()

    @property
    def state_string(self):
        return f"Loop: {self.loop_number}/{self.total_loops_per_trial}; " f"Bad loops: {self.bad_turn_counter}"

    async def print_state(self, refresh_rate: float = 0.2):
        while True:
            if self._delay_countdown_task:
                await self._delay_countdown_task
            print(self.state_string, end="\r")
            await asyncio.sleep(refresh_rate)

    def initiate_countdown(self, seconds: int):
        self._delay_countdown_task = asyncio.create_task(self.countdown(seconds))

    def stop_countdown_prematurely(self):
        self._delay_countdown_task.cancel()
        clear_console()

    async def countdown(self, seconds: int = 10):
        logger.debug(f"Starting countdown timer for {self}")

        start = datetime.now()
        for _ in tqdm(range(seconds)):
            await asyncio.sleep(1.0)
        stop = datetime.now()

        logger.debug(f"Countdown was finalised. Number of seconds {(total_time := stop - start)}")
        self.countdown_timings.append((start, stop, total_time))

    def localize_loop_func(self, location: int, timestamp: datetime):
        raise NotImplementedError
