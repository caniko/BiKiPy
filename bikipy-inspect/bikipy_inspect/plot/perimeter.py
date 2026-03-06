from __future__ import annotations

from typing import Any

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from bikipy_inspect.manifest import InspectionManifest, PerimeterSpec
from bikipy_inspect.plot.generic import plot_ellipse


def plot_perimeter_on_ax(
    ax: Axes,
    perimeter: PerimeterSpec,
    color: Any = "g",
    meters_per_pixel: float = 0.0,
    as_pixels: bool = False,
) -> None:
    """Draw a perimeter shape on the given axis.

    Supports circle, rectangle, polygon, triangle, and radial_maze.
    """
    scale = 1.0 / meters_per_pixel if (as_pixels and meters_per_pixel > 0) else 1.0

    if perimeter.shape == "circle":
        center = (perimeter.params["center_x"] * scale, perimeter.params["center_y"] * scale)
        radius = perimeter.params["radius"] * scale
        plot_ellipse(ax, center, radius, color=color)

    elif perimeter.shape == "rectangle":
        cx = perimeter.params["center_x"] * scale
        cy = perimeter.params["center_y"] * scale
        w = perimeter.params["width"] * scale
        h = perimeter.params["height"] * scale
        rect = plt.Rectangle(
            (cx - w / 2, cy - h / 2), w, h,
            fill=False, edgecolor=color, linewidth=2,
        )
        ax.add_patch(rect)

    elif perimeter.shape in ("polygon", "triangle"):
        verts = np.array(perimeter.params["vertices"]) * scale
        closed = np.vstack([verts, verts[0]])
        ax.plot(closed[:, 0], closed[:, 1], color=color, linewidth=2)

    elif perimeter.shape == "radial_maze":
        # Draw center polygon
        center_verts = np.array(perimeter.params["center_vertices"]) * scale
        closed = np.vstack([center_verts, center_verts[0]])
        ax.plot(closed[:, 0], closed[:, 1], color=color, linewidth=2)
        # Draw each arm
        for arm_verts in perimeter.params["arms"]:
            arm = np.array(arm_verts) * scale
            closed_arm = np.vstack([arm, arm[0]])
            ax.plot(closed_arm[:, 0], closed_arm[:, 1], color=color, linewidth=1.5, linestyle="--")


def subplot_with_perimeter(
    manifest: InspectionManifest,
    perimeter_label: str | None = None,
    nrows: int = 1,
    ncols: int = 1,
    figsize: tuple[float, float] | None = None,
    **subplot_kwargs,
) -> tuple[Figure, Any]:
    """Create a figure with perimeter(s) pre-drawn on all axes.

    If video resolution is available, sizes the figure accordingly.
    """
    if figsize is None:
        w, h = manifest.video.resolution
        if w > 0 and h > 0:
            aspect = h / w
            figsize = (10, 10 * aspect * nrows / ncols)
        else:
            figsize = (10, 8)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, **subplot_kwargs)

    perimeters = (
        [manifest.get_perimeter(perimeter_label)]
        if perimeter_label
        else manifest.perimeters
    )

    for ax in np.array(axes).flatten():
        for p in perimeters:
            plot_perimeter_on_ax(ax, p, meters_per_pixel=manifest.settings.meters_per_pixel)

    return fig, axes
