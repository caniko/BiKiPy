import copy
import subprocess
from functools import lru_cache
from logging import getLogger
from pathlib import Path, PurePath
from typing import Any, Union

import cv2
import numpy as np
from matplotlib import pyplot as plt

from bikipy.core.typing import NDArrayFp64

logger = getLogger(__name__)


def dict_deepmerge(source: dict, destination: dict) -> dict:
    """
    run me with nosetests --with-doctest file.py

    >>> a = { 'first' : { 'all_rows' : { 'pass' : 'dog', 'number' : '1' } } }
    >>> b = { 'first' : { 'all_rows' : { 'fail' : 'cat', 'number' : '5' } } }
    >>> merge(b, a) == { 'first' : { 'all_rows' : { 'pass' : 'dog', 'fail' : 'cat', 'number' : '5' } } }
    True
    """
    for key, value in source.items():
        if isinstance(value, dict):
            # get node or create one
            node = destination.setdefault(key, {})
            dict_deepmerge(value, node)
        else:
            destination[key] = value

    return destination


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


def get_git_root():
    return Path(
        subprocess.Popen(["git", "rev-parse", "--show-toplevel"], stdout=subprocess.PIPE)
        .communicate()[0]
        .rstrip()
        .decode("utf-8")
    )
