from typing import Sequence, SupportsFloat

import matplotlib.pyplot as plt
import numpy as np


def order_parallelogram_corners(perimeter_corners: Sequence):
    perimeter_corners = np.asarray(perimeter_corners)
    argsorted_x, argsorted_y = np.argsort(perimeter_corners.T, axis=1)

    # Order corners with respect to perimeter
    vertical_side_a = argsorted_x[:2]
    # vertical_side_b = argsorted_x[2:]

    horizontal_side_a = argsorted_y[2:]
    horizontal_side_b = argsorted_y[:2]

    down_left, down_right, up_left, up_right = None, None, None, None
    for corner in vertical_side_a:
        if corner in horizontal_side_b:
            assert not down_left
            down_left = perimeter_corners[corner]
            (down_right,) = perimeter_corners[
                horizontal_side_b[horizontal_side_b != corner]
            ]

        elif corner in horizontal_side_a:
            assert not up_left
            up_left = perimeter_corners[corner]
            (up_right,) = perimeter_corners[
                horizontal_side_a[horizontal_side_a != corner]
            ]

    assert np.all((result := np.array((down_left, down_right, up_right, up_left))))
    return result


def expand_parallelogram(
    perimeter_corners: Sequence, offset: SupportsFloat, inspect: bool = False
):
    offset = float(offset)
    down_left, down_right, up_left, up_right = order_parallelogram_corners(
        perimeter_corners
    )

    off_up_left = (up_left[0] - offset, up_left[1] + offset)
    off_down_left = (down_left[0] - offset, down_left[1] - offset)
    off_down_right = (down_right[0] + offset, down_right[1] - offset)
    off_up_right = (up_right[0] + offset, up_right[1] + offset)

    if inspect:
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
            plt.scatter(*np.array(point).T)

        plt.legend(
            (
                "down_left",
                "down_right",
                "up_left",
                "up_right",
                "off_down_left",
                "off_down_right",
                "off_up_left",
                "off_up_right",
            ),
            bbox_to_anchor=(1.04, 0.5),
            loc="center left",
        )

        plt.show()

    return off_down_left, off_down_right, off_up_left, off_up_right
