from logging import getLogger
from typing import TYPE_CHECKING, Optional

import numpy as np
from matplotlib import pyplot as plt, patches
from matplotlib.axes import Axes
from pydantic import validate_arguments
from pydantic_numpy.dtype import NDArrayBool, NDArrayFp64

from bikipy.utils.plot.color import cmap

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
    ax: Axes = None,
    inspect_pixels: bool = False,
    video: Optional["VideoMetadata"] = None,
    **plot_kwargs,
):
    coordinates = video.prepare_coordinates_for_plotting(coordinates, inspect_pixels)

    ax.plot(*coordinates.T, color=cmap(len(coordinates)), **plot_kwargs)

    return ax


@validate_arguments(config={"arbitrary_types_allowed": True})
def plot_ellipse(center: tuple[float, float], radius: tuple[float, float] | float, ax: Axes = None) -> Axes:
    if ax is None:
        fig, ax = plt.subplots()

    if isinstance(radius, tuple) and np.isclose(radius[0], radius[1]):
        radius = radius[0]

    if isinstance(radius, float):
        circle = plt.Circle(center, radius, fill=False)
        ax.add_artist(circle)
    elif isinstance(radius, tuple):
        ellipse = patches.Ellipse(center, *radius, edgecolor="g", facecolor="none")
        ax.add_patch(ellipse)
    else:
        msg = f"Provided radius has invalid type: {type(radius)}"
        raise ValueError(msg)

    ax.scatter(*center, marker=",")

    return ax
