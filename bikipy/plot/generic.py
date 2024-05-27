import os
from logging import getLogger
from pathlib import Path
from typing import TYPE_CHECKING, Any, Generator, Iterator, Literal, Optional, Sequence

import numpy as np
from matplotlib import colors, patches
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from pydantic import ConfigDict, validate_call
from pydantic_numpy.typing import Np1DArrayBool, Np2DArrayFp64

from bikipy._constant import INSPECT_FIG_FILE_FORMAT
from bikipy.math.discrete import boolean_index_truth_sequence_start_end
from bikipy.utils.misc import int_file_stem_incrementor

if TYPE_CHECKING:
    from bikipy.core.video import VideoMetadata


logger = getLogger(__file__)


def generic_figure_finalization(
    inspection_fig_output_path: Path,
    potential_dir: Optional[str] = None,
    potential_label: Optional[str] = None,
    inspect_fig_file_format: Literal[".svgz", ".jpg"] = INSPECT_FIG_FILE_FORMAT,
    close: bool = True,
) -> None:
    if inspection_fig_output_path.is_file():
        file_path = inspection_fig_output_path / potential_label if potential_label else inspection_fig_output_path

        logger.debug(f"Ensuring that {file_path.parent} directory exists")
        os.makedirs(file_path.parent, exist_ok=True)

        if file_path.stem.split("-")[0].isdigit():
            file_path = int_file_stem_incrementor(file_path)

        file_path = file_path.with_suffix(inspect_fig_file_format)

    else:
        assert inspection_fig_output_path.is_dir()

        if potential_dir:
            inspection_fig_output_path = inspection_fig_output_path / potential_dir

        logger.debug(f"Creating directory {inspection_fig_output_path} for inspection figures")
        os.makedirs(inspection_fig_output_path, exist_ok=True)

        stem = f"1-{potential_label}" if potential_label else "1"
        file_path = int_file_stem_incrementor((inspection_fig_output_path / stem).with_suffix(inspect_fig_file_format))

    logger.debug(f"Saving inspection figure, file path: {file_path}")

    plt.tight_layout()
    plt.savefig(file_path)

    if close:
        plt.close("all")


def color_map_by_number(number: int, cmap: Any = plt.cm.cool) -> Iterator:
    return cmap(np.linspace(0, 1, number))


def boolean_index_colormap(
    boolean_index: Sequence[bool], cmap_true: Any = plt.cm.cool, cmap_false: Any = plt.cm.Wistia
) -> Generator:
    length = len(boolean_index)
    for state, color_true, color_false in zip(
        boolean_index, color_map_by_number(length, cmap_true), color_map_by_number(length, cmap_false)
    ):
        yield color_true if state else color_false


def ax_hue_plot_coordinate_with_boolean_index(
    ax,
    boolean_index: Np1DArrayBool,
    coordinates: Np2DArrayFp64,
    plot_non_confinement: bool = False,
    plot_line: bool = False,
) -> None:
    if plot_line:
        x, y = coordinates.T

        for start, end in boolean_index_truth_sequence_start_end(boolean_index):
            for line_x, line_y, color in zip(
                zip(x[start:end], x[start + 1 : end]),
                zip(y[start:end], y[start + 1 : end]),
                color_map_by_number(end - start - 1),
            ):
                ax.plot(line_x, line_y, color=color, linewidth=3.0)

        if plot_non_confinement:
            for start, end in boolean_index_truth_sequence_start_end(~boolean_index):
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


def ax_hue_plot_coordinates(ax: Axes, coordinates: Np2DArrayFp64, marker: str = "x") -> None:
    for color, coord in zip(color_map_by_number(len(coordinates)), coordinates):
        ax.scatter(*coord.T, color=color, marker=marker)


def ax_hue_plot_coordinate_pair_as_lines(ax: Axes, coordinates_a: Np2DArrayFp64, coordinates_b: Np2DArrayFp64) -> None:
    paired_coordiantes = np.dstack((coordinates_a, coordinates_b))
    for color, pair in zip(color_map_by_number(len(paired_coordiantes)), paired_coordiantes):
        ax.plot(*pair, color=color)


def plot_coordinates(
    ax: Axes,
    coordinates: Np2DArrayFp64,
    coordinates_as_pixels: bool = False,
    video: Optional["VideoMetadata"] = None,
    color: Any = None,
    **plot_kwargs,
) -> None:
    if video:
        coordinates = video.prepare_coordinates_for_plotting(coordinates, coordinates_as_pixels)
    ax.plot(*coordinates.T, color=color, **plot_kwargs)


@validate_call(config=ConfigDict(arbitrary_types_allowed=True))
def plot_ellipse(ax: Axes, center: tuple[float, float], radius: tuple[float, float] | float, color: Any = None) -> None:
    if isinstance(radius, tuple) and np.isclose(radius[0], radius[1]):
        radius = radius[0]

    color_to_assign = color or "g"

    if isinstance(radius, float):
        circle = plt.Circle(center, radius, fill=False, color=colors.to_rgba(color_to_assign))
        ax.add_artist(circle)

    elif isinstance(radius, tuple):
        ellipse = patches.Ellipse(center, *radius, edgecolor=color_to_assign, facecolor="none")
        ax.add_patch(ellipse)

    else:
        msg = f"Provided radius has invalid type: {type(radius)}"
        raise ValueError(msg)

    ax.scatter(*center, marker=",")
