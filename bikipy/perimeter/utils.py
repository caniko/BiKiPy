from typing import Optional

import numpy as np
from pydantic import FilePath

from bikipy.utils.misc import read_makesense_point_csv


def distance_between_two_perimeters(perimeter_a, perimeter_b):
    return np.linalg.norm(perimeter_a.centroid - perimeter_b.centroid)


def get_coco_array_from_path_or_array(
    metadata_path: Optional[FilePath] = None,
    coco_array: Optional[np.ndarray] = None,
):
    msg = "metadata_path and coco_array are defined mutually exclusive"
    if metadata_path and np.any(coco_array):
        raise ValueError(msg)

    if metadata_path:
        result = read_makesense_point_csv(metadata_path)
    elif np.any(coco_array):
        result = coco_array
    else:
        raise ValueError(msg)

    assert np.any(result)
    return result
