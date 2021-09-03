import asyncio
from datetime import datetime
from logging import getLogger

import numpy as np
from tqdm import tqdm

from bikipy.behaviour.base import BaseTrial

try:
    import zmq
    import zmq.asyncio
except ImportError as e:
    msg = "You need to install BiKiPy[live] to use the infinity maze module"
    raise ImportError(msg) from e

logger = getLogger(__name__)


class LiveTrial(BaseTrial):
    def __init__(self, *args, **kwargs):
        self.countdown_timings = []

        self._zmq_context = None
        self._socket = None

    def generate_zmq_context(self, socket_address: str = "tcp://*:5555"):
        self._zmq_context = zmq.asyncio.Context()
        self._socket = self._zmq_context.socket(zmq.PULL)
        self._socket.bind(str(socket_address))

    async def live_localize(
        self,
        data_separator_delimiter: str = ",",
        data_delimiter: str = " ",

    ):
        self.generate_zmq_context()

        try:
            while True:
                message = await self._socket.recv_string()

                coordinate_str, timestamp_str = message.split(data_separator_delimiter)

                location = self.detect_confined_perimeter(
                    np.array(coordinate_str.split(data_delimiter), dtype=np.float32)
                )
                logger.debug(
                    f"Coordinates: {coordinate_str}\n"
                    f"Timestamp: {timestamp_str}\n"
                    f"Location: {location}"
                )
                await self.localize_loop_func(
                    location,
                    datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S.%f")
                )
        except KeyboardInterrupt:
            request_save = input(
                "Would you like to save the results from the experiment? Y/n\n"
            ).lower()
            if not request_save or request_save == "y":
                self.save()
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

    async def localize_loop_func(self, location: int, timestamp: datetime):
        raise NotImplementedError
