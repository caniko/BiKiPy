from functools import lru_cache
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
from math import sqrt
from pydantic import validate_arguments

from bikipy.core.typing import NDArrayFp64
from bikipy.feature.angle import clockwise_angel_2d


@validate_arguments
def find_cathetus_from_similar_triangle_with_hypotenuse_points_from_original_triangle_and_length_of_the_target_triangle(
    hypotenuse_start: NDArrayFp64, hypotenuse_end: NDArrayFp64, inspect: bool = False
):
    """
    We utilize the diagonal of rectangle to derive the components of the two axes on the 2D image.
    We derive both the meters and pixels of the diagonal, and use the Pythagoras theorem for this:

    https://www.reddit.com/r/askmath/comments/j1bvfj/getting_catheti_from_hypotenuse_and_catheti_ratio/?utm_source=share&utm_medium=web2x&context=3
    hypotenuse h and the ratio, r, of x and y in a right triangle.

    x/y=r -> x=y*r

    h**2 = x**2 + y**2
    h**2 = y**2 + (y*r)**2
    h**2 = y**2 * (1 + r**2)

    y = sqrt( h**2 / (1 + r**2) )
    x = sqrt( h**2 - y**2 )

    We can override the hypotenuse length if we want to calculate
    """
    pixel_ab_vector = np.abs(hypotenuse_end - hypotenuse_start)
    pixel_x, pixel_y = pixel_ab_vector
    pixel_xy_ratio = pixel_x / pixel_y  # a-b intersects on the origin

    meter_y = sqrt(self.meter_length**2.0 / (1.0 + pixel_xy_ratio**2.0))
    meter_x = sqrt(self.meter_length**2.0 - meter_y**2.0)

    if inspect:
        pixel_x_vector = np.array([pixel_x, 0.0])
        plt.plot(*np.vstack([[0.0, 0.0], pixel_x_vector]).T, label="cathetus_x")

        pixel_y_vector = np.array([0.0, pixel_y])
        plt.plot(*np.vstack([[0.0, 0.0], pixel_y_vector]).T, label="cathetus_y")

        plt.plot(*np.vstack([pixel_x_vector, pixel_y_vector]).T, label="hypotenuse")

        plt.legend()
        plt.show()

    return pixel_x, pixel_y


def clockwise_argsort_points(points: NDArrayFp64):
    points = np.asarray(points)
    assert points.ndim == 2
    centroid = np.mean(points, axis=0)

    return np.argsort(clockwise_angel_2d((0.0, 1.0), points - centroid))


def clockwise_sort_points(points: NDArrayFp64, inspect: bool = False):
    # Sort from top-right point
    points = np.asarray(points)
    result = points[clockwise_argsort_points(points)]

    if inspect:
        fig, ax = plt.subplots()
        for point in result:
            ax.scatter(*point)
        plt.legend([f"result_{i}" for i in range(1, len(points) + 1)])
        plt.show()

    return result


@lru_cache
def clockwise_sort_perimeter_centroids(perimeters: Sequence):
    return [perimeters[i] for i in clockwise_argsort_points([perimeter.centroid for perimeter in perimeters])]


@lru_cache
def expand_bikipy_perimeter(perimeter, *args, **kwargs):
    return np.array(expand_rectangle(perimeter.vertices_in_meters, *args, **kwargs))


def expand_rectangle(
    perimeter_vertices: Sequence,
    offset: Sequence[float] | float,
    y_inverted: bool = False,
    inspect: bool = False,
    as_array: bool = False,
):
    if isinstance(offset, float):
        x_offset = offset = float(offset)
        y_offset = -offset if y_inverted else offset
    else:
        x_offset, y_offset = offset
        if y_inverted:
            y_offset = -y_offset

    up_right, down_right, down_left, up_left = clockwise_sort_points(perimeter_vertices)

    off_down_left = (down_left[0] - x_offset, down_left[1] - y_offset)
    off_down_right = (down_right[0] + x_offset, down_right[1] - y_offset)
    off_up_right = (up_right[0] + x_offset, up_right[1] + y_offset)
    off_up_left = (up_left[0] - x_offset, up_left[1] + y_offset)

    if inspect:
        fig, axes = plt.subplots(2, 1)
        fig.gca().invert_yaxis()

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

        plt.show()

    result = (off_down_left, off_down_right, off_up_right, off_up_left)
    return np.array(result) if as_array else result


cathetus_from_hypotenuse([5, 5], [10, 10], True)
