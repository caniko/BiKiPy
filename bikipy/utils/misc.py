import copy
import subprocess
from functools import lru_cache
from logging import getLogger
from pathlib import Path, PurePath
from typing import Any, Union

import cv2
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from numpy import ndarray
from pydantic import FilePath
from pydantic_numpy import NDArray

logger = getLogger(__name__)


def read_image(image: Any, imread_flagg: Any = None):
    if isinstance(image, (PurePath, str)):
        image_path = Path(image).resolve()
        assert image_path.exists(), image_path
        image = cv2.imread(str(image_path), flags=imread_flagg)
    else:
        assert isinstance(image, ndarray), f"image must be either path or NDArray, but got:\n{image}"

    return image


def get_reference_point_from_array(array: NDArray):
    return np.array(array[1:3], dtype=float)


def generic_inspection_finalization(inspect):
    if isinstance(inspect, bool):
        plt.show()
    elif isinstance(inspect, str) or isinstance(inspect, PurePath):
        plt.savefig(inspect)


def seek_next_file_index(filepath: Union[PurePath, str]) -> PurePath:
    if not (original_filepath := Path(filepath)).exists():
        return original_filepath.with_stem(f"{1:04d}_{original_filepath.stem}")

    new_filepath = copy.copy(original_filepath)
    i = 2
    while new_filepath.exists():
        new_filepath = filepath.with_stem(f"{i:04d}_{original_filepath.stem}")
        i += 1
    return new_filepath


def rise_to_n_levels(columns, n_levels: int):
    """
    Used to equate two pandas dataframes in terms of their column levels before merge

    :param columns:
    :param n_levels:
    :return:
    """
    column_array = np.array(columns)
    if len(column_array.shape) == 1:
        column_array = np.expand_dims(column_array, 1)

    if n_levels < column_array.shape[1]:
        msg = "Can not reduce the number of levels that are natively defined" "in index"
        raise ValueError(msg)

    return to_tuple(
        np.concatenate(
            (
                column_array,
                [["" for _ in range(n_levels - column_array.shape[1])]] * len(column_array),
            ),
            axis=1,
        )
    )


@lru_cache
def generic_multi_indexer(*basis_labels):
    number_of_levels = len(basis_labels)

    def result(category: Any, level: int):
        if level < number_of_levels:
            msg = f"At least {number_of_levels} levels; provided={level}"
            raise ValueError(msg)
        levels_to_add = ["" for _ in range(level - number_of_levels)]
        return [(category, basis_label, *levels_to_add) for basis_label in basis_labels]

    return result


def directory_incrementor(path: Path):
    path = Path(path)
    i = 2
    while path.exists():
        path = path.with_stem(path.stem + f"_{i}")
        i += 1
    return path


def clear_console():
    """
    https://stackoverflow.com/a/65343640/9793651
    :return:
    """
    print("\033c\033[3J", end="")


def to_tuple(array: NDArray):
    return tuple(map(tuple, array))


def get_git_root():
    return Path(
        subprocess.Popen(["git", "rev-parse", "--show-toplevel"], stdout=subprocess.PIPE)
        .communicate()[0]
        .rstrip()
        .decode("utf-8")
    )
