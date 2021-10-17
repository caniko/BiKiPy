import copy
from logging import getLogger
from pathlib import Path, PurePath
from typing import Any, Union

import cv2
import pandas as pd
from numpy import ndarray

from bikipy.utils.typing import Path_typing

logger = getLogger(__name__)


def read_image(image: Any, imread_flagg: Any = None):
    if isinstance(image, str) or isinstance(image, PurePath):
        image_path = Path(image).resolve()
        assert image_path.exists(), image_path
        image = cv2.imread(str(image_path), flags=imread_flagg)
    else:
        assert isinstance(
            image, ndarray
        ), f"image must be either path or np.ndarray, but got:\n{image}"

    return image


def read_makesense_point_csv(coco_path: Path_typing):
    return pd.read_csv(
        coco_path,
        header=None,
        # names=["x1", "y1", "x2", "y2", "filename", "img_x", "img_y"],
    ).to_numpy()


def seek_next_file_index(filepath: Union[PurePath, str]) -> PurePath:
    if not (original_filepath := Path(filepath)).exists():
        return original_filepath

    new_filepath = copy.copy(original_filepath)
    i = 2
    while new_filepath.exists():
        new_filepath = filepath.with_suffix(str(i))
        i += 1
    return new_filepath


def clear_console():
    """
    https://stackoverflow.com/a/65343640/9793651
    :return:
    """
    print("\033c\033[3J", end="")
