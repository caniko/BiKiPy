from typing import Any, SupportsInt, SupportsFloat, Sequence
from PIL import Image

import matplotlib.pyplot as plt
import numpy as np

from bikipy.compute.midpoint import compute_midpoint
from bikipy.utils.math import unit_vector


def define_object(guiding_image_path: Any, n: SupportsInt = 3):
    try:
        img = Image.open(guiding_image_path)
    except:
        img = guiding_image_path

    plt.imshow(img)

    sides = plt.ginput(n=n, timeout=0)
    return sides


def generate_object_borders(sides: Sequence, distance: SupportsFloat):
    if isinstance(sides, np.ndarray):
        sides_as_list = sides.tolist()
    else:
        sides_as_list = list(sides)

    point_a = [
        np.array(sublist) for sublist in [sides_as_list[-1]] + sides_as_list[:-1]
    ]
    point_b = [np.array(sublist) for sublist in sides_as_list[1:] + [sides_as_list[0]]]

    side_roots = compute_midpoint(point_a, point_b)

    outward_vectors = np.asanyarray(sides) - side_roots
    outward_unit_vectors = np.apply_along_axis(
        lambda x: unit_vector(x), 1, outward_vectors
    )
    return float(distance) * outward_unit_vectors
