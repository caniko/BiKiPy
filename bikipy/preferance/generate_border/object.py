from typing import Any, Union, SupportsInt, SupportsFloat, Sequence
from PIL import Image

import matplotlib.pyplot as plt
import numpy as np

from bikipy.compute.midpoint import compute_midpoint
from bikipy.utils.math import unit_vector


def define_object(guiding_image_path: Any, n: SupportsInt = 4):
    img = Image.open(guiding_image_path)
    return plt.ginput(img, n=n)


def generate_object_borders(sides: Sequence, distance: SupportsFloat):
    sides = np.asanyarray(sides)
    side_roots = np.array([
        compute_midpoint(
            sides[i - 1], sides[i + 1 if i + 1 != sides.shape[0] else 0]
        ) for i in range(sides.shape[0]-1)
    ])
    outward_vectors = sides - side_roots
    outward_unit_vectors = np.apply_along_axis(
        lambda x: unit_vector(x), 1, outward_vectors
    )
    return float(distance) * outward_unit_vectors
