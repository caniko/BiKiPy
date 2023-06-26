import numpy as np
from pydantic_numpy import NDArray


def np_sum_int(array: NDArray) -> int:
    return int(np.sum(array))
