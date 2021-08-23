import numpy as np

try:
    import keyboard
    import zmq
    import zmq.asyncio
except ImportError as e:
    msg = "You need to install BiKiPy[live] to use the infinity maze module"
    raise ImportError(msg) from e

from bikipy.behaviour.base import BaseTrial
from bikipy.perimeter.base import Perimeter2D


class InfinityMaze(BaseTrial):
    def __init__(
        self,
        delay_perimeter: Perimeter2D,
        stem_perimeter: Perimeter2D,
        choice_perimeter: Perimeter2D,
        reward_perimeter_left: Perimeter2D,
        reward_perimeter_right: Perimeter2D,
        return_perimeter: Perimeter2D,
        live: bool = False,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.delay_perimeter = delay_perimeter
        self.stem_perimeter = stem_perimeter
        self.choice_perimeter = choice_perimeter
        self.reward_perimeter_left = reward_perimeter_left
        self.reward_perimeter_right = reward_perimeter_right
        self.return_perimeter = return_perimeter

        self.perimeters = {
            "delay": self.delay_perimeter,
            "stem": self.stem_perimeter,
            "choice": self.choice_perimeter,
            "reward_left": self.reward_perimeter_left,
            "reward_right": self.reward_perimeter_right,
            "return": self.return_perimeter
        }

        self.live = live
        self.zmq_context = zmq.asyncio.Context() if self.live else None

        self.coordinate_sequence = None

    async def localize(self):
        if not self.live:
            msg = "live was not set to True during initialisation"
            raise ValueError(msg)

        socket = self.zmq_context.socket(zmq.PULL)
        socket.bind("tcp://*:5555")

        coordinate_sequence = []
        while True:
            message = await socket.recv_string()

            location = self.detect_confined_perimeter(
                np.array(message.split(" "), dtype=np.float32)
            )
