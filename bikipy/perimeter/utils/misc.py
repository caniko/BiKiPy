from logging import getLogger
from typing import Optional

import numpy as np
from pydantic import FilePath
from pydantic_numpy.typing import Np2DArrayFp64

from bikipy.utils.makesense import read_makesense_point

logger = getLogger(__file__)


def get_coco_array_from_path_or_array(
    metadata_path: Optional[FilePath],
    coco_array: Optional[Np2DArrayFp64],
):
    msg = "metadata_path and coco_array are defined mutually exclusive"
    if metadata_path and np.any(coco_array):
        raise ValueError(msg)

    if metadata_path:
        result = read_makesense_point(metadata_path)
    elif np.any(coco_array):
        result = coco_array
    else:
        raise ValueError(msg)

    assert np.any(result)
    return result
