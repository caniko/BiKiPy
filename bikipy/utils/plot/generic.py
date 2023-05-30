from logging import getLogger
from typing import TYPE_CHECKING, Any, Optional

import numpy as np
from matplotlib import patches
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from pydantic import validate_arguments
from pydantic_numpy.dtype import NDArrayBool, NDArrayFp64

from bikipy.utils.math.discrete import boolean_index_truth_sequence_start_end
from bikipy.utils.plot.color import cmap

if TYPE_CHECKING:
    from bikipy.core.video import VideoMetadata


logger = getLogger(__file__)


def ax_plot_coordinate_with_boolean_index(
    ax, boolean_index: NDArrayBool, coordinates: NDArrayFp64, plot_false: bool = False, plot_line: bool = False
) -> None:
    if plot_line:
        x, y = coordinates.T

        for start, end, length in boolean_index_truth_sequence_start_end(boolean_index):
            ax.plot(x[start:end], y[start:end], c=plt.cm.winter(np.linspace(0, 1, length)), linewidth=3.0)

        if plot_false:
            for start, end, length in boolean_index_truth_sequence_start_end(~boolean_index):
                ax.plot(x[start:end], y[start:end], c=plt.cm.Wistia(np.linspace(0.1, 1, length)), linewidth=3.0)

    else:
        ax.scatter(*coordinates[boolean_index].T, label="Valid", color="dodgerblue")
        if plot_false:
            ax.scatter(*coordinates[~boolean_index].T, label="Invalid", color="crimson")


def plot_coordinates(
    coordinates: NDArrayFp64,
    ax: Axes = None,
    inspect_pixels: bool = False,
    video: Optional["VideoMetadata"] = None,
    color: Any = None,
    **plot_kwargs,
):
    if not ax:
        if video:
            _fig, ax = video.subplot()
        else:
            print("Video not provided")
            _fig, ax = plt.subplots()

    if video:
        coordinates = video.prepare_coordinates_for_plotting(coordinates, inspect_pixels)

    ax.plot(*coordinates.T, color=color or cmap(len(coordinates)), **plot_kwargs)

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
