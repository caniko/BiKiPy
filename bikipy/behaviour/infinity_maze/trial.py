from bikipy.behaviour.base import BaseTrial


class InfinityMaze(BaseTrial):
    def __init__(self, coordinate_sequence: dict, *args, **kwargs):
        super().__init__(*args, **kwargs)
