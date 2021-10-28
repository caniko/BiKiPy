import copy
import os
from logging import getLogger
from pathlib import Path, PurePath
from typing import Any, Union

import cv2
import pandas as pd
from matplotlib import pyplot as plt
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
        # dtype={0: str, 1: float, 2: float, 3: str, 4: int, 5: int}
        # names=["x1", "y1", "x2", "y2", "filename", "img_x", "img_y"],
    ).to_numpy()


def generic_inspection_finalization(inspect, category: str):
    if isinstance(inspect, bool):
        plt.show()
    elif isinstance(inspect, str) or isinstance(inspect, PurePath):
        inspect = Path(inspect).resolve()
        if not inspect.parent.exists():
            os.makedirs(inspect.parent)
        plt.savefig(seek_next_file_index(inspect / category / f"{category}.jpg"))


def seek_next_file_index(filepath: Union[PurePath, str]) -> PurePath:
    if not (original_filepath := Path(filepath)).exists():
        return original_filepath.with_suffix(f"{1:04d}")

    new_filepath = copy.copy(original_filepath)
    i = 2
    while new_filepath.exists():
        new_filepath = filepath.with_suffix(f"{i:04d}")
        i += 1
    return new_filepath


def clear_console():
    """
    https://stackoverflow.com/a/65343640/9793651
    :return:
    """
    print("\033c\033[3J", end="")
