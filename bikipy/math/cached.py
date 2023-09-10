from functools import lru_cache

import numpy as np


@lru_cache
def cached_deg2rad(deg: float) -> float:
    return np.deg2rad(deg)


@lru_cache
def meters2pixels(meters: float, pixels_per_meter: float) -> float:
    return meters * pixels_per_meter
