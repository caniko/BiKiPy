from typing import Any

import numpy as np
from matplotlib import pyplot as plt

from bikipy.core.typing import NDArrayFp64


def plot_circle(center: NDArrayFp64, radius: NDArrayFp64 | float, ax: Any = None):
    angles = np.linspace(0, 2 * np.pi, 200)

    result = center + radius * np.array([np.cos(angles), np.sin(angles)]).T

    if ax is None:
        fig, ax = plt.subplots()

    ax.plot(*result.T)

    return ax
