from __future__ import annotations

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from bikipy_inspect.manifest import InspectionManifest
from bikipy_inspect.plot.generic import (
    ax_hue_plot_coordinate_with_boolean_index,
    boolean_index_truth_sequences,
)
from bikipy_inspect.plot.perimeter import plot_perimeter_on_ax


def plot_heuristic_summary(manifest: InspectionManifest, figsize: tuple[float, float] = (12, 5)) -> Figure:
    """Bar chart summarizing all heuristic results (percentage of frames True)."""
    if not manifest.heuristics:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "No heuristic results", ha="center", va="center", transform=ax.transAxes)
        return fig

    names = [h.name for h in manifest.heuristics]
    percentages = [h.percentage for h in manifest.heuristics]
    seconds = [h.seconds for h in manifest.heuristics]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    bars = ax1.barh(names, percentages, color="steelblue")
    ax1.set_xlabel("% of frames")
    ax1.set_title("Heuristic Results (percentage)")
    ax1.set_xlim(0, 100)

    ax2.barh(names, seconds, color="darkorange")
    ax2.set_xlabel("Seconds")
    ax2.set_title("Heuristic Results (duration)")

    fig.tight_layout()
    return fig


def plot_heuristic_timeline(
    manifest: InspectionManifest,
    heuristic_name: str,
    figsize: tuple[float, float] = (14, 4),
) -> Figure:
    """Timeline plot showing when a heuristic is True across frames."""
    boolean_index = manifest.get_boolean_index(heuristic_name)
    fps = manifest.video.fps

    fig, ax = plt.subplots(figsize=figsize)

    # Draw True spans as colored rectangles
    sequences = boolean_index_truth_sequences(boolean_index)
    for start, end in sequences:
        t_start = start / fps
        t_end = end / fps
        ax.axvspan(t_start, t_end, alpha=0.4, color="steelblue")

    total_seconds = len(boolean_index) / fps
    ax.set_xlim(0, total_seconds)
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.set_xlabel("Time (s)")
    ax.set_title(f"{heuristic_name} — timeline")

    meta = next(h for h in manifest.heuristics if h.name == heuristic_name)
    ax.text(
        0.99, 0.95,
        f"{meta.seconds:.1f}s ({meta.percentage:.1f}%)",
        transform=ax.transAxes, ha="right", va="top",
        fontsize=10, color="dimgray",
    )

    fig.tight_layout()
    return fig


def plot_heuristic_on_coordinates(
    manifest: InspectionManifest,
    heuristic_name: str,
    label: str,
    perimeter_label: str | None = None,
    plot_non_confinement: bool = True,
    plot_line: bool = False,
    figsize: tuple[float, float] = (10, 10),
) -> Figure:
    """Plot coordinates color-coded by a heuristic boolean result, with optional perimeter overlay."""
    boolean_index = manifest.get_boolean_index(heuristic_name)
    coordinates = manifest.get_coordinates(label)

    fig, ax = plt.subplots(figsize=figsize)
    ax_hue_plot_coordinate_with_boolean_index(
        ax, boolean_index, coordinates,
        plot_non_confinement=plot_non_confinement,
        plot_line=plot_line,
    )

    if perimeter_label:
        p = manifest.get_perimeter(perimeter_label)
        plot_perimeter_on_ax(ax, p, meters_per_pixel=manifest.settings.meters_per_pixel)
    else:
        for p in manifest.perimeters:
            plot_perimeter_on_ax(ax, p, meters_per_pixel=manifest.settings.meters_per_pixel)

    ax.set_title(f"{heuristic_name} — {label}")
    ax.legend()
    ax.set_aspect("equal")
    fig.tight_layout()
    return fig
