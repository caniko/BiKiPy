import numpy as np
from pydantic_numpy.typing import NpNDArray


def np_sum_int(array: NpNDArray) -> int:
    return int(np.sum(array))


def seconds_to_frames(seconds: float, fps: float) -> int:
    return round(seconds * fps)
