from typing import Any, Optional

import numpy as np
import seaborn as sb
from matplotlib import pyplot as plt
from pydantic_numpy.dtype import NDArrayBool, NDArrayFp64

from bikipy import runtime_settings
from bikipy.core.video import VideoMetadata
from bikipy.feature.angle import angle_from_a_to_b
from bikipy.perimeter.base import SinglePerimeter
from bikipy.utils.math.vector import unit_vector


def ray_direction_filter_circle_triangle(
    perimeter: SinglePerimeter,
    ray_travel_direction_point: NDArrayFp64,
    ray_start_point: NDArrayFp64,
    max_radians: float,
    inspect: bool = False,
    **inspect_kwargs,
) -> NDArrayBool:
    ray_vectors = ray_travel_direction_point - ray_start_point

    closest_points_on_edges = perimeter.closest_point_on_edge_to_coordinates(ray_travel_direction_point)
    vector_to_closest_point_on_edge = perimeter.vector_to_closest_point_on_edge(ray_travel_direction_point)

    direction_point_is_closer_than_start_point = np.linalg.norm(
        closest_points_on_edges - ray_travel_direction_point, axis=1
    ) <= np.linalg.norm(closest_points_on_edges - ray_start_point, axis=1)

    angle_from_normal_to_ray = angle_from_a_to_b(vector_to_closest_point_on_edge, ray_vectors)

    result = direction_point_is_closer_than_start_point & (np.abs(angle_from_normal_to_ray) <= max_radians)

    if inspect:
        ray_inspection_plot(
            perimeter, result, unit_vector(ray_vectors) * 0.025, ray_travel_direction_point, **inspect_kwargs
        )

    return result


def ray_inspection_plot(
    perimeter: SinglePerimeter,
    result: NDArrayFp64,
    ray_vectors: NDArrayFp64,
    ray_travel_direction_point: NDArrayFp64,
    inspect_video: Optional[VideoMetadata] = None,
    inspect_pixels: bool = False,
    manual_ax: Any = None,
):
    if manual_ax is None:
        sb.set_theme(style="darkgrid")
        fig, ax = plt.subplots(dpi=500)
    else:
        ax = manual_ax

    ray_travel_direction_point = inspect_video.prepare_coordinates_for_plotting(
        ray_travel_direction_point, inspect_pixels
    )
    ray_vectors = inspect_video.prepare_coordinates_for_plotting(ray_vectors, inspect_pixels)

    if inspect_pixels:
        inspect_video.upscaled_video.ax_ticks_metric_to_pixel(ax)

    perimeter.plot(inspect_pixels=inspect_pixels, ax=ax)
    ax.set_title("Gaze direction filter", fontsize=inspect_video.upscaled_video.plotting_title_font_size)

    quiver_kwargs = {
        "angles": "xy",
        "scale_units": "dots",
        "scale": 1.0,
        "alpha": runtime_settings.matplotlib_scatter_alpha,
    }

    ax.quiver(*ray_travel_direction_point[result].T, *ray_vectors[result].T, label="Valid", color="b", **quiver_kwargs)

    not_result = ~result
    ax.quiver(
        *ray_travel_direction_point[not_result].T,
        *ray_vectors[not_result].T,
        label="Invalid",
        color="r",
        **quiver_kwargs,
    )

    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.025),
        fancybox=True,
        ncol=2,
        fontsize=inspect_video.upscaled_video.plotting_default_font_size,
    )

    if not manual_ax:
        plt.tight_layout()
        plt.show()

    return ax
