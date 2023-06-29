from logging import getLogger
from typing import TYPE_CHECKING, Any, Optional

import numpy as np
from matplotlib import colors, patches
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from pydantic import validate_arguments
from pydantic_numpy.dtype import NDArrayBool, NDArrayFp64

from bikipy.utils.math.discrete import boolean_index_truth_sequence_start_end

if TYPE_CHECKING:
    from bikipy.core.video import VideoMetadata


logger = getLogger(__file__)


def ax_plot_coordinate_with_boolean_index(
    ax,
    boolean_index: NDArrayBool,
    coordinates: NDArrayFp64,
    plot_non_confinement: bool = False,
    plot_line: bool = False,
) -> None:
    if plot_line:
        x, y = coordinates.T

        for start, end in boolean_index_truth_sequence_start_end(boolean_index):
            for line_x, line_y, color in zip(
                zip(x[start:end], x[start + 1 : end]),
                zip(y[start:end], y[start + 1 : end]),
                plt.cm.winter(np.linspace(0, 1, end - start - 1)),
            ):
                ax.plot(line_x, line_y, c=color, linewidth=3.0)

        if plot_non_confinement:
            for start, end in boolean_index_truth_sequence_start_end(~boolean_index):
                for line_x, line_y, color in zip(
                    zip(x[start:end], x[start + 1 : end]),
                    zip(y[start:end], y[start + 1 : end]),
                    plt.cm.Wistia(np.linspace(0, 1, end - start - 1)),
                ):
                    ax.plot(line_x, line_y, c=color, linewidth=3.0)

    else:
        ax.scatter(*coordinates[boolean_index].T, label="Valid", color="dodgerblue")
        if plot_non_confinement:
            ax.scatter(*coordinates[~boolean_index].T, label="Invalid", color="crimson")


def plot_coordinates(
    ax: Axes,
    coordinates: NDArrayFp64,
    coordinates_as_pixels: bool = False,
    video: Optional["VideoMetadata"] = None,
    color: Any = None,
    **plot_kwargs,
) -> None:
    if video:
        coordinates = video.prepare_coordinates_for_plotting(coordinates, coordinates_as_pixels)
    ax.plot(*coordinates.T, color=color, **plot_kwargs)


@validate_arguments(config={"arbitrary_types_allowed": True})
def plot_ellipse(ax: Axes, center: tuple[float, float], radius: tuple[float, float] | float, color: Any = None) -> None:
    if isinstance(radius, tuple) and np.isclose(radius[0], radius[1]):
        radius = radius[0]

    color = color or "g"

    if isinstance(radius, float):
        circle = plt.Circle(center, radius, fill=False, color=colors.to_rgba(color) if color else None)
        ax.add_artist(circle)
    elif isinstance(radius, tuple):
        ellipse = patches.Ellipse(center, *radius, edgecolor=color, facecolor="none")
        ax.add_patch(ellipse)
    else:
        msg = f"Provided radius has invalid type: {type(radius)}"
        raise ValueError(msg)

    ax.scatter(*center, marker=",")
