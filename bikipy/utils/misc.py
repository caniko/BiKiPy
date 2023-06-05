import copy
import os
import subprocess
from collections import defaultdict
from functools import partial
from logging import getLogger
from pathlib import Path, PurePath
from typing import Optional, Union

import openpyxl
from odf import opendocument
from odf.table import Table
from pydantic import DirectoryPath, FilePath

logger = getLogger(__name__)


def sheet_names_from_path(path_to_workbook: FilePath) -> list[str]:
    if path_to_workbook.suffix == ".ods":
        # odfpy really needs documentation. Had to reverse-engineer this: https://pastebin.com/Xp9dqvRq
        return [
            sheet.getAttribute("name")
            for sheet in opendocument.load(path_to_workbook).spreadsheet.getElementsByType(Table)
        ]
    return openpyxl.load_workbook(path_to_workbook, read_only=True).sheetnames


def dict_deepmerge(source: dict, destination: dict, assert_no_endpoint_intersection: bool = False) -> dict:
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
            assert not (key in destination and assert_no_endpoint_intersection)
            destination[key] = value

    return destination


def seek_next_file_index(filepath: Union[PurePath, str]) -> PurePath:
    if not (original_filepath := Path(filepath)).exists():
        return original_filepath.with_stem(f"{1:04d}_{original_filepath.stem}")

    new_filepath = copy.copy(original_filepath)
    i = 2
    while new_filepath.exists():
        new_filepath = filepath.with_stem(f"{i:04d}_{original_filepath.stem}")
        i += 1
    return new_filepath


def int_file_stem_incrementor(starting_filename: Path, delimiter: str = "-"):
    while starting_filename.exists():
        split_stem = starting_filename.stem.split(delimiter)

        try:
            int_id = int(split_stem[0]) + 1
        except ValueError as e:
            msg = "The filename to increment must have a digit in the beginning that must be split with a dash"
            raise ValueError(msg) from e

        starting_filename = starting_filename.with_stem(f"{int_id}-{'-'.join(split_stem[1:])}")
    return starting_filename


def clear_console():
    """
    https://stackoverflow.com/a/65343640/9793651
    :return:
    """
    print("\033c\033[3J", end="")


def current_path_or_arg_path(project_directory: Optional[DirectoryPath]):
    return Path(project_directory or os.curdir)


def get_git_root():
    return Path(
        subprocess.Popen(["git", "rev-parse", "--show-toplevel"], stdout=subprocess.PIPE)
        .communicate()[0]
        .rstrip()
        .decode("utf-8")
    )


defaultdict_dict_factory = partial(defaultdict, dict)
