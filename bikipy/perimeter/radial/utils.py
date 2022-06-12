import numpy as np

from bikipy.core.typing import NDArrayFp64


def plot_circle(center: NDArrayFp64, radius: float, ax):
    angles = np.linspace(0, 2 * np.pi, 200)

    x = center[0] + radius * np.cos(angles)
    y = center[1] + radius * np.sin(angles)

    ax.plot(x, y)
    return ax
