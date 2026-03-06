from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Iterator, Literal

import numpy as np
from matplotlib import colors, patches
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from bikipy_inspect.plot.color import color_map_by_number

INSPECT_FIG_FILE_FORMAT = ".svgz"
MINIMUM_FIG_DPI = 300


def save_figure(
    fig: Figure,
    output_path: Path,
    fmt: Literal[".svgz", ".jpg", ".png", ".pdf"] = INSPECT_FIG_FILE_FORMAT,
    close: bool = True,
) -> Path:
    """Save a figure to disk, creating directories as needed."""
    os.makedirs(output_path.parent, exist_ok=True)
    save_path = output_path.with_suffix(fmt)
    fig.savefig(save_path, bbox_inches="tight", dpi=MINIMUM_FIG_DPI)
    if close:
        plt.close(fig)
    return save_path


def boolean_index_truth_sequences(boolean_index: np.ndarray) -> list[tuple[int, int]]:
    """Find contiguous True sequences, returning (start, end) pairs."""
    if len(boolean_index) == 0:
        return []

    sequences = []
    in_sequence = False
    start = 0

    for i, val in enumerate(boolean_index):
        if val and not in_sequence:
            start = i
            in_sequence = True
        elif not val and in_sequence:
            sequences.append((start, i))
            in_sequence = False

    if in_sequence:
        sequences.append((start, len(boolean_index)))

    return sequences


def ax_hue_plot_coordinate_with_boolean_index(
    ax: Axes,
    boolean_index: np.ndarray,
    coordinates: np.ndarray,
    plot_non_confinement: bool = False,
    plot_line: bool = False,
) -> None:
    """Plot coordinates with color-coding based on a boolean index.

    When plot_line is True, draws colored line segments for True sequences.
    Otherwise, draws scatter points (blue=True, red=False).
    """
    if plot_line:
        x, y = coordinates.T

        for start, end in boolean_index_truth_sequences(boolean_index):
            for line_x, line_y, color in zip(
                zip(x[start:end], x[start + 1 : end]),
                zip(y[start:end], y[start + 1 : end]),
                color_map_by_number(end - start - 1),
            ):
                ax.plot(line_x, line_y, color=color, linewidth=3.0)

        if plot_non_confinement:
            for start, end in boolean_index_truth_sequences(~boolean_index):
                for line_x, line_y, color in zip(
                    zip(x[start:end], x[start + 1 : end]),
                    zip(y[start:end], y[start + 1 : end]),
                    color_map_by_number(end - start - 1, cmap=plt.cm.Wistia),
                ):
                    ax.plot(line_x, line_y, color=color, linewidth=3.0)
    else:
        ax.scatter(*coordinates[boolean_index].T, label="Valid", color="dodgerblue")
        if plot_non_confinement:
            ax.scatter(*coordinates[~boolean_index].T, label="Invalid", color="crimson")


def ax_hue_plot_coordinates(ax: Axes, coordinates: np.ndarray, marker: str = "x") -> None:
    """Plot coordinates with hue gradient."""
    for color, coord in zip(color_map_by_number(len(coordinates)), coordinates):
        ax.scatter(*coord.T, color=color, marker=marker)


def ax_hue_plot_coordinate_pair_as_lines(
    ax: Axes, coordinates_a: np.ndarray, coordinates_b: np.ndarray
) -> None:
    """Connect paired coordinates with colored lines."""
    paired = np.dstack((coordinates_a, coordinates_b))
    for color, pair in zip(color_map_by_number(len(paired)), paired):
        ax.plot(*pair, color=color)


def plot_coordinates(
    ax: Axes,
    coordinates: np.ndarray,
    color: Any = None,
    **plot_kwargs,
) -> None:
    """Plot a coordinate trajectory on an axis."""
    ax.plot(*coordinates.T, color=color, **plot_kwargs)


def plot_ellipse(
    ax: Axes,
    center: tuple[float, float],
    radius: tuple[float, float] | float,
    color: Any = None,
) -> None:
    """Draw a circle or ellipse on an axis."""
    if isinstance(radius, tuple) and np.isclose(radius[0], radius[1]):
        radius = radius[0]

    color_to_assign = color or "g"

    if isinstance(radius, float):
        circle = plt.Circle(center, radius, fill=False, color=colors.to_rgba(color_to_assign))
        ax.add_artist(circle)
    elif isinstance(radius, tuple):
        ellipse = patches.Ellipse(center, *radius, edgecolor=color_to_assign, facecolor="none")
        ax.add_patch(ellipse)

    ax.scatter(*center, marker=",")
