import asyncio
from collections import Sequence as collections_Sequence
from datetime import datetime
from logging import getLogger
from pathlib import PurePath
from types import Union
from typing import Sequence

import numpy as np
from tqdm import tqdm

from bikipy.behaviour.base import BaseTrial
from bikipy.utils.store import RangeDict

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


class LiveTrial(BaseTrial):
    def __init__(
        self,
        *args,
        delay_timings: Sequence[Union[float, int]],
        delay_timings_trial_count: Union[int, Sequence[int]],
        **kwargs,
    ):
        super().__init__(*args, _live=True, **kwargs)
        if not hasattr(self, "save_root"):
            msg = "save_root must be defined for saving the trial data after conclusion"
            raise ValueError(msg)

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
                delay_timings_trial_count for _ in range(len(self.delay_timings))
            )

        self.node_sequence = []
        self.bad_loop_record = {}

        self._loop_number_vs_delay_time = (
            RangeDict(
                {
                    i: delay_time
                    for i, delay_time in zip(
                        self.delay_timings_trial_count, self.delay_timings
                    )
                },
                allow_less_than_first_key=0,
            )
            if delay_timings
            else {}
        )

        self.live_expose_metrics = []
        self.countdown_timings = []

        self._zmq_context = None
        self._socket = None

    def generate_zmq_context(self, socket_address: str = "tcp://*:5555"):
        self._zmq_context = zmq.asyncio.Context()
        self._socket = self._zmq_context.socket(zmq.PULL)
        self._socket.bind(str(socket_address))

    def application(self):
        event_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(event_loop)

        try:
            event_loop.run_until_complete(self.localization_loop())
        except KeyboardInterrupt:
            logger.info("Localization loop interrupted with keypress")

        request_save = input(
            "Would you like to save the results from the experiment? Y/n\n"
        ).lower()
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
                logger.debug(
                    f"Coordinates: {coordinate_str}\n"
                    f"Timestamp: {timestamp_str}\n"
                    f"Location: {location}"
                )
                self.localize_loop_func(
                    location, datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S.%f")
                )
        finally:
            self._socket.close()

    async def countdown(self, seconds: int = 10):
        start = datetime.now()
        logger.debug(f"Starting countdown timer for {self}")
        for _ in tqdm(range(seconds)):
            await asyncio.sleep(1.0)
        stop = datetime.now()
        logger.debug(
            f"Countdown was finalised. Number of seconds {(total_time := stop - start)}"
        )
        self.countdown_timings.append((start, stop, total_time))

    def localize_loop_func(self, location: int, timestamp: datetime):
        raise NotImplementedError
