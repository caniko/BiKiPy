from functools import lru_cache

import numpy as np
import seaborn
from matplotlib import cm


@lru_cache
def cmap(n: int):
    return tuple(cm.cool(x) for x in np.linspace(0.0, 1.0, n))


@lru_cache
def make_color_map(n_colors: int):
    return seaborn.color_palette("dark", n_colors)
