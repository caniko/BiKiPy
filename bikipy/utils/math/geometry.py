import math
from functools import lru_cache
from math import sqrt
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as nt
from pydantic import validate_arguments
from pydantic_numpy.dtype import NDArrayFp64

from bikipy.feature.angle import clockwise_angel_2d
from bikipy.utils.plot.inspect import generic_inspection_finalization

if TYPE_CHECKING:
    from bikipy.perimeter.base import Perimeter


def normalize_hypotenuse_to_origin(hypotenuse_start: NDArrayFp64, hypotenuse_end: NDArrayFp64):
    return np.abs(hypotenuse_end - hypotenuse_start)


@validate_arguments
def cathetus_from_similar_triangle_with_hypotenuse_points_from_original_triangle_and_length_of_the_target_triangle(
    cathetus_a: NDArrayFp64, cathetus_b: NDArrayFp64, similar_hypotenuse_length: float, inspect: bool = False
):
    """
    We utilize the diagonal of rectangle to derive the components of the two axes on the 2D image.
    We derive both the meters and pixels of the diagonal, and use the Pythagoras theorem for this:

    https://www.reddit.com/r/askmath/comments/j1bvfj/getting_catheti_from_hypotenuse_and_catheti_ratio/?utm_source=share&utm_medium=web2x&context=3
    hypotenuse h and the ratio, r, of a and b in a right triangle.

    a/b=r -> a=b*r

    h**2 = a**2 + b**2
    h**2 = b**2 + (b*r)**2
    h**2 = b**2 * (1 + r**2)

    b = sqrt( h**2 / (1 + r**2) )
    a = sqrt( h**2 - b**2 )

    We can override the hypotenuse length if we want to calculate
    """
    ab_ratio = cathetus_a / cathetus_b  # a-b intersects on the origin

    similar_b = sqrt(similar_hypotenuse_length**2.0 / (1.0 + ab_ratio**2.0))
    similar_a = sqrt(similar_hypotenuse_length**2.0 - similar_b**2.0)

    if inspect:
        fig, axes = plt.subplots(1, 2)
        for ax, (a, b) in zip(axes, ((cathetus_a, cathetus_b), (similar_a, similar_b))):
            vector_a = np.array([a, 0.0])
            ax.plot(*np.vstack([[0.0, 0.0], vector_a]).T, label="a")

            vector_b = np.array([0.0, b])
            ax.plot(*np.vstack([[0.0, 0.0], vector_b]).T, label="b")

            ax.plot(*np.vstack([vector_a, vector_b]).T, label="hypotenuse")

        axes[0].set_title("Original")
        axes[1].set_title("Similar")

        plt.legend()
        plt.show()

    return similar_a, similar_b


def meter_per_pixel_from_diagonal(diagonal_a: NDArrayFp64, diagonal_b: NDArrayFp64, length_meters: float):
    pixel_x, pixel_y = normalize_hypotenuse_to_origin(diagonal_b, diagonal_a)
    (
        meter_x,
        meter_y,
    ) = cathetus_from_similar_triangle_with_hypotenuse_points_from_original_triangle_and_length_of_the_target_triangle(
        pixel_x, pixel_y, length_meters
    )

    result = np.array([meter_x / pixel_x, meter_y / pixel_y])
    if math.isclose(*result, rel_tol=10**-5):
        return np.mean(result)
    return result


def clockwise_argsort_points(points: NDArrayFp64):
    points = np.asarray(points)
    assert points.ndim == 2
    centroid_meters = np.mean(points, axis=0)

    return np.argsort(clockwise_angel_2d((0.0, 1.0), points - centroid_meters))


@validate_arguments
def clockwise_sort_points(points: NDArrayFp64, inspect: bool = False) -> nt.NDArray:
    # Sort from top-right point
    result = points[clockwise_argsort_points(points)]

    if inspect:
        fig, ax = plt.subplots()
        for point in result:
            ax.scatter(*point)
        plt.legend([f"result_{i}" for i in range(1, len(points) + 1)])
        plt.show()

    return result


@lru_cache
def clockwise_sort_perimeter_centroids(perimeters: Sequence["Perimeter"]):
    return [perimeters[i] for i in clockwise_argsort_points([perimeter.centroid_meters for perimeter in perimeters])]


@lru_cache
def expand_bikipy_perimeter(perimeter, *args, **kwargs):
    return np.array(expand_rectangle(perimeter.vertices_in_meters, *args, **kwargs))


def expand_rectangle(
    perimeter_vertices: Sequence,
    x_offset: float = 0,
    y_offset: float = 0,
    y_inverted: bool = False,
    inspection_fig_output_path: Optional[Path] = None,
) -> np.ndarray:
    if y_inverted and y_offset:
        y_offset = -y_offset

    up_right, down_right, down_left, up_left = clockwise_sort_points(perimeter_vertices)

    off_down_left = (down_left[0] - x_offset, down_left[1] - y_offset)
    off_down_right = (down_right[0] + x_offset, down_right[1] - y_offset)
    off_up_right = (up_right[0] + x_offset, up_right[1] + y_offset)
    off_up_left = (up_left[0] - x_offset, up_left[1] + y_offset)

    if inspection_fig_output_path:
        fig, axes = plt.subplots(2, 1)

        fig.suptitle("Rectangle expansion")
        axes[0].set_title("Scattered")
        axes[1].set_title("Line")
        fig.tight_layout()

        for point in (
            down_left,
            down_right,
            up_left,
            up_right,
            off_down_left,
            off_down_right,
            off_up_left,
            off_up_right,
        ):
            axes[0].scatter(*np.array(point).T)

        box = axes[0].get_position()
        axes[0].set_position([box.x0, box.y0, box.width * 0.65, box.height])

        axes[0].legend(
            (
                "down-left",
                "down-right",
                "up-left",
                "up-right",
                "off-down-left",
                "off-down-right",
                "off-up-left",
                "off-up-right",
            ),
            bbox_to_anchor=(1.04, 0.5),
            loc="center left",
        )

        axes[1].plot(
            (down_left[0], down_right[0]),
            (down_left[1], down_right[1]),
            "o-",
            (down_right[0], up_right[0]),
            (down_right[1], up_right[1]),
            "o-",
            (up_right[0], up_left[0]),
            (up_right[1], up_left[1]),
            "o-",
            (up_left[0], down_left[0]),
            (up_left[1], down_left[1]),
            "o-",
            (off_down_left[0], off_down_right[0]),
            (off_down_left[1], off_down_right[1]),
            "o-",
            (off_down_right[0], off_up_right[0]),
            (off_down_right[1], off_up_right[1]),
            "o-",
            (off_up_right[0], off_up_left[0]),
            (off_up_right[1], off_up_left[1]),
            "o-",
            (off_up_left[0], off_down_left[0]),
            (off_up_left[1], off_down_left[1]),
            "o-",
        )

        box = axes[1].get_position()
        axes[1].set_position([box.x0, box.y0, box.width * 0.675, box.height])
        axes[1].legend(
            (
                "down_l-r",
                "down-up",
                "up_r-l",
                "up-down",
                "off-down_l-r",
                "off-down-up",
                "off-up_r-l",
                "off-up-down",
            ),
            bbox_to_anchor=(1.04, 0.5),
            loc="center left",
        )

        generic_inspection_finalization(inspection_fig_output_path, potential_dir="rectangle_expansion")

    result = (off_down_left, off_down_right, off_up_right, off_up_left)

    return np.array(result)
