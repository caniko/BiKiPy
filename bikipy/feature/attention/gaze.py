from typing import Optional, Any

import numpy as np
from matplotlib import pyplot as plt
import seaborn as sb

from bikipy import MATPLOTLIB_SCATTER_ALPHA
from bikipy.core.typing import NDArrayFp64, NDArrayBool
from bikipy.core.video import VideoMetadata, convert_meters_to_pixels
from bikipy.feature.angle import angle_from_a_to_b
from bikipy.perimeter.base import AnyPerimeter
from bikipy.utils.collection_utils import evenly_spaced_indices


def gaze_direction_filter_circle_triangle(
    perimeter: AnyPerimeter,
    gaze_travel_direction_point: NDArrayFp64,
    gaze_start_point: NDArrayFp64,
    max_radians: float,
    inspect: bool = False,
    **inspect_kwargs
) -> NDArrayBool:
    gaze_vector = gaze_travel_direction_point - gaze_start_point

    closest_points_on_edges = perimeter.closest_point_on_edge_to_coordinates(gaze_travel_direction_point)
    vector_to_closest_point_on_edge = perimeter.vector_to_closest_point_on_edge(gaze_travel_direction_point)

    direction_point_is_closer_than_start_point = np.linalg.norm(
        closest_points_on_edges - gaze_travel_direction_point, axis=1
    ) < np.linalg.norm(closest_points_on_edges - gaze_start_point, axis=1)

    angle_from_normal_to_gaze = angle_from_a_to_b(vector_to_closest_point_on_edge, gaze_vector)

    result = direction_point_is_closer_than_start_point & (np.abs(angle_from_normal_to_gaze) <= max_radians)

    if inspect:
        _gaze_inspection_plot(**inspect_kwargs)

    return result


def _gaze_inspection_plot(
    perimeter: AnyPerimeter,
    result: NDArrayFp64,
    gaze_vector: NDArrayFp64,
    gaze_travel_direction_point: NDArrayFp64,
    vector_to_closest_point_on_edge: NDArrayFp64,
    closest_points_on_edges: NDArrayFp64,
    inspect_video: Optional[VideoMetadata] = None,
    inspect_pixels: bool = False,
    inspect_edge_normals: bool = False,
    inspect_vectors: bool = False,
    inspection_ax: Any = None,
):
    if inspection_ax is None:
        sb.set_theme(style="darkgrid")
        fig, ax = plt.subplots(dpi=500)
    else:
        ax = inspection_ax

    if inspect_pixels:
        gaze_travel_direction_point = convert_meters_to_pixels(gaze_travel_direction_point, inspect_video)
        if inspect_edge_normals:
            closest_points_on_edges = convert_meters_to_pixels(closest_points_on_edges, inspect_video)

    perimeter.plot(inspect_pixels=inspect_pixels, ax=ax)
    ax.set_title("Gaze direction filter")

    quiver_kwargs = {
        "angles": "xy",
        # "scale_units": "xy",
        "scale": 1.0,
        "alpha": MATPLOTLIB_SCATTER_ALPHA,
    }

    ax.quiver(
        *gaze_travel_direction_point[result].T, *gaze_vector[result].T, label="Valid", color="b", **quiver_kwargs
    )

    not_result = ~result
    ax.quiver(
        *gaze_travel_direction_point[not_result].T,
        *gaze_vector[not_result].T,
        label="Invalid",
        color="r",
        **quiver_kwargs,
    )

    if inspect_edge_normals:
        ax.quiver(
            *closest_points_on_edges.T,
            *vector_to_closest_point_on_edge.T,
            label="EdgeNormals",
            color="g",
            **quiver_kwargs,
        )

    if inspect_vectors:
        number_of_points = 5
        with sb.color_palette("Spectral", n_colors=number_of_points):
            for i in evenly_spaced_indices(gaze_travel_direction_point, number_of_points):
                ax.plot(*np.vstack((closest_points_on_edges[i], gaze_travel_direction_point[i])).T)

    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.025), fancybox=True, ncol=2)

    if not inspection_ax:
        plt.tight_layout()
        plt.show()

    return ax
