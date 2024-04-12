from functools import lru_cache

import numpy as np
import seaborn
from matplotlib import cm


@lru_cache
def make_color_map(n_colors: int):
    return seaborn.color_palette("cool", n_colors)
