from functools import lru_cache

import seaborn


@lru_cache
def make_color_map(n_colors: int):
    return seaborn.color_palette("cool", n_colors)
