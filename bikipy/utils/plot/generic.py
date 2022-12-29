from logging import getLogger
from typing import TYPE_CHECKING, Any, Optional

import numpy as np
from matplotlib import pyplot as plt
from pydantic_numpy.dtype import NDArrayBool, NDArrayFp64

from bikipy.utils.plot import cmap

if TYPE_CHECKING:
    from bikipy.core.video import VideoMetadata


logger = getLogger(__file__)


def ax_plot_coordinate_with_boolean_index(
    ax, boolean_index: NDArrayBool, coordinates: NDArrayFp64, plot_line: bool = False
) -> None:
    from bikipy.utils.plot import BOTTOM_LEGEND_KWARGS

    length = len(boolean_index)
    assert length == len(coordinates)

    if plot_line:
        inside_colors = plt.cm.winter(np.linspace(0, 1, length))
        # outside_colors = plt.cm.Wistia(np.linspace(0, 1, length))

        plot_colors = plt.cm.Wistia(np.linspace(0, 1, length))
        plot_colors[boolean_index] = inside_colors[boolean_index]

        for color, point_a, point_b in zip(plot_colors, coordinates, coordinates[1:]):
            ax.plot(*np.vstack((point_a, point_b)).T, c=color, linewidth=3.0)
    else:
        ax.scatter(*coordinates[boolean_index].T, label="Inside", color="dodgerblue")
        ax.scatter(*coordinates[~boolean_index].T, label="Outside", color="crimson")

        ax.legend(**BOTTOM_LEGEND_KWARGS)


def plot_coordinates(
    coordinates: NDArrayFp64,
    ax: Any = None,
    inspect_pixels: bool = False,
    video: Optional["VideoMetadata"] = None,
    **plot_kwargs,
):
    coordinates = video.prepare_coordinates_for_plotting(coordinates, inspect_pixels)

    ax.plot(*coordinates.T, color=cmap(len(coordinates)), **plot_kwargs)

    return ax


def plot_circle(center: NDArrayFp64, radius: NDArrayFp64 | float, ax: Any = None):
    angles = np.linspace(0, 2 * np.pi, 200)

    result = center + radius * np.array([np.cos(angles), np.sin(angles)]).T

    if ax is None:
        fig, ax = plt.subplots()

    ax.plot(*result.T)
    ax.scatter(*center, marker=",")

    return ax
