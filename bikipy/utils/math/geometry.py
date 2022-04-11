from collections.abc import Sequence
from functools import lru_cache

import matplotlib.pyplot as plt
import numpy as np

from bikipy.feature.angle import clockwise_angel_2d


def clockwise_argsort_points(points: Sequence):
    points = np.asarray(points)
    assert points.ndim == 2
    centroid = np.mean(points, axis=0)

    return np.argsort(clockwise_angel_2d((0.0, 1.0), points - centroid))


def clockwise_sort_points(points: Sequence, inspect: bool = False):
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


clockwise_sort_points(((0, 0), (1, 0), (1, 1), (0, 1)), inspect=True)


@lru_cache
def clockwise_sort_perimeter_centroids(perimeters: Sequence):
    return [perimeters[i] for i in clockwise_argsort_points([perimeter.centroid for perimeter in perimeters])]


@lru_cache
def expand_bikipy_perimeter(perimeter, *args, **kwargs):
    return np.array(expand_parallelogram(perimeter.corners, *args, **kwargs))


def expand_parallelogram(
    perimeter_corners: Sequence,
    offset: float,
    y_inverted: bool = False,
    inspect: bool = False,
    as_array: bool = False,
):
    x_offset = offset = float(offset)
    y_offset = -offset if y_inverted else offset

    up_right, down_right, down_left, up_left = clockwise_sort_points(perimeter_corners)

    off_down_left = (down_left[0] - x_offset, down_left[1] - y_offset)
    off_down_right = (down_right[0] + x_offset, down_right[1] - y_offset)
    off_up_right = (up_right[0] + x_offset, up_right[1] + y_offset)
    off_up_left = (up_left[0] - x_offset, up_left[1] + y_offset)

    if inspect:
        fig, axes = plt.subplots(2, 1)
        fig.gca().invert_yaxis()

        fig.suptitle("Parallelogram expansion")
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
