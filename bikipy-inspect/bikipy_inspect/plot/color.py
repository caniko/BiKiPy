from __future__ import annotations

from functools import lru_cache
from typing import Any, Generator, Iterator, Sequence

import numpy as np
import seaborn
from matplotlib import pyplot as plt


@lru_cache
def make_color_palette(n_colors: int):
    return seaborn.color_palette("cool", n_colors)


def color_map_by_number(number: int, cmap: Any = plt.cm.cool) -> Iterator:
    return cmap(np.linspace(0, 1, number))


def boolean_index_colormap(
    boolean_index: Sequence[bool],
    cmap_true: Any = plt.cm.cool,
    cmap_false: Any = plt.cm.Wistia,
) -> Generator:
    length = len(boolean_index)
    for state, color_true, color_false in zip(
        boolean_index,
        color_map_by_number(length, cmap_true),
        color_map_by_number(length, cmap_false),
    ):
        yield color_true if state else color_false
