from functools import lru_cache

import numpy as np


@lru_cache
def cached_deg2rad(deg: float) -> float:
    return np.deg2rad(deg)
