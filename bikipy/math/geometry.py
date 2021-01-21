from typing import Sequence, SupportsFloat

import matplotlib.pyplot as plt
import numpy as np


def order_parallelogram_corners(sides: Sequence):
    sides = np.asanyarray(sides)
    argsorted_x, argsorted_y = np.argsort(sides.T, axis=1)

    # Order corners with respect to perimeter
    vertical_side_a = argsorted_x[:2]
    # vertical_side_b = argsorted_x[2:]

    horizontal_side_a = argsorted_y[2:]
    horizontal_side_b = argsorted_y[:2]

    down_left, down_right, up_left, up_right = None, None, None, None
    for corner in vertical_side_a:
        if corner in horizontal_side_b:
            assert not down_left
            down_left = sides[corner]
            (down_right,) = sides[horizontal_side_b[horizontal_side_b != corner]]

        elif corner in horizontal_side_a:
            assert not up_left
            up_left = sides[corner]
            (up_right,) = sides[horizontal_side_a[horizontal_side_a != corner]]

    assert np.all((result := np.array((down_left, down_right, up_right, up_left))))
    return result


def expand_parallelogram(sides: Sequence, offset: SupportsFloat, inspect: bool = False):
    offset = float(offset)
    down_left, down_right, up_left, up_right = order_parallelogram_corners(sides)

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
            )
        )

        plt.show()

    return off_down_left, off_down_right, off_up_left, off_up_right
