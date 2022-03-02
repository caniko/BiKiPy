from typing import Sequence

import numpy as np


def plot_circle(center: Sequence, radius: float, ax):
    angles = np.linspace(0, 2 * np.pi, 200)

    x = center[0] + radius * np.cos(angles)
    y = center[1] + radius * np.sin(angles)

    ax.plot(x, y)
    return ax
