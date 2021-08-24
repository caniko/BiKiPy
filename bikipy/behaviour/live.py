try:
    import zmq
    import zmq.asyncio
except ImportError as e:
    msg = "You need to install BiKiPy[live] to use the infinity maze module"
    raise ImportError(msg) from e

import numpy as np

from bikipy.behaviour.base import BaseTrial


class LiveTrial(BaseTrial):
    def __init__(self, coordinate_sequence: dict):
        super().__init__(coordinate_sequence)

        self.zmq_context = zmq.asyncio.Context()

    async def localize(self, delimiter: str = " "):
        socket = self.zmq_context.socket(zmq.PULL)
        socket.bind("tcp://*:5555")

        try:
            while True:
                message = await socket.recv_string()

                location = self.detect_confined_perimeter(
                    np.array(message.split(delimiter), dtype=np.float32)
                )
                self._localize_loop_func(location)
        except KeyboardInterrupt:
            socket.close()

    @staticmethod
    def _localize_loop_func(location: np.ndarray):
        raise NotImplementedError
