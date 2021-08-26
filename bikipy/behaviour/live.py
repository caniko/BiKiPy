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
    def __init__(self, coordinate_sequence: dict):
        super().__init__(coordinate_sequence)

        self.zmq_context = zmq.asyncio.Context()
        self.countdown_timings = []

    async def localize(
        self, delimiter: str = " ", socket_address: str = "tcp://*:5555"
    ):
        socket = self.zmq_context.socket(zmq.PULL)
        socket.bind(str(socket_address))

        try:
            while True:
                message = await socket.recv_string()

                location = self.detect_confined_perimeter(
                    np.array(message.split(delimiter), dtype=np.float32)
                )
                asyncio.create_task(self.localize_loop_func(location))
        except KeyboardInterrupt:
            socket.close()

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

    async def localize_loop_func(self, location: int):
        raise NotImplementedError
