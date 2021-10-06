import itertools
from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np

from bikipy.feature.angle import counterclockwise_angel_2d


def order_polygon_corners(perimeter_corners: Sequence, inspect: bool = False):
    perimeter_corners = np.asarray(perimeter_corners)
    centroid = np.mean(perimeter_corners, axis=0)

    centroid_corner_vectors = perimeter_corners - centroid
    result = perimeter_corners[argsort_counterclockwise(centroid_corner_vectors)]

    if inspect:
        fig, ax = plt.subplots()
        for point in result:
            ax.scatter(*point)
        plt.legend([f"result_{i}" for i in range(1, len(perimeter_corners) + 1)])
        plt.show()

    return result


def argsort_counterclockwise(sequence: Sequence):
    return np.argsort(counterclockwise_angel_2d((-1.0, 0.0), sequence))


def expand_parallelogram(
    perimeter_corners: Sequence,
    offset: float,
    y_inverted: bool = True,
    inspect: bool = False,
):
    offset = float(offset)
    down_left, down_right, up_right, up_left = order_polygon_corners(perimeter_corners)

    x_offset = offset
    y_offset = -offset if y_inverted else offset

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

    return off_down_left, off_down_right, off_up_right, off_up_left
