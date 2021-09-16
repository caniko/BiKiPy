from pathlib import PurePath
from typing import Union

import numpy as np

from bikipy.perimeter import TriangularPerimeter
from bikipy.perimeter.base import PolygonalPerimeter


def y_maze_perimeters_can_method(
    a_apex_line: np.ndarray,
    b_apex_line: np.ndarray,
    c_apex_line: np.ndarray,
    center_coco_path: Union[PurePath, str, None] = None,
    center_object: Union[TriangularPerimeter, None] = None,
):
    center_object = center_object or PolygonalPerimeter.from_coco(center_coco_path)
    if not center_object:
        msg = "Either center_object or center_coco_path has to be defined"
        raise ValueError(msg)
