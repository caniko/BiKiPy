import copy
import os
from logging import getLogger
from pathlib import Path, PurePath
from typing import Any, Union

import cv2
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from numpy import ndarray
from numpy.typing import NDArray

from bikipy.utils.typing import PathTyping

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


def read_makesense_point_csv(coco_path: PathTyping):
    return pd.read_csv(
        coco_path,
        header=None,
        # dtype={0: str, 1: float, 2: float, 3: str, 4: int, 5: int}
        # names=["x1", "y1", "x2", "y2", "filename", "img_x", "img_y"],
    ).to_numpy()


def get_reference_point_from_array(array: np.ndarray):
    return np.array(array[1:3], dtype=float)


def generic_inspection_finalization(inspect, category: str):
    if isinstance(inspect, bool):
        plt.show()
    elif isinstance(inspect, str) or isinstance(inspect, PurePath):
        root = Path(inspect).resolve() / category
        if not root.exists():
            os.makedirs(root)
        plt.savefig(seek_next_file_index(root / f"{category}.jpg"))


def seek_next_file_index(filepath: Union[PurePath, str]) -> PurePath:
    if not (original_filepath := Path(filepath)).exists():
        return original_filepath.with_stem(f"{1:04d}_{original_filepath.stem}")

    new_filepath = copy.copy(original_filepath)
    i = 2
    while new_filepath.exists():
        new_filepath = filepath.with_stem(f"{i:04d}_{original_filepath.stem}")
        i += 1
    return new_filepath


def clear_console():
    """
    https://stackoverflow.com/a/65343640/9793651
    :return:
    """
    print("\033c\033[3J", end="")


def to_tuple(array: NDArray):
    return tuple(map(tuple, array))


def boolean_index_islands(boolean_index: np.ndarray):
    
