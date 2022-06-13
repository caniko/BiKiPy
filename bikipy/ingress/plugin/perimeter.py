import shutil
from logging import getLogger
from pathlib import Path

import plyer
import yaml
from pydantic import DirectoryPath, FilePath, validate_arguments

from bikipy.ingress.utils.io import (
    get_perimeter_directory_path,
    get_project_settings_path,
    load_settings,
)

logger = getLogger(__name__)


@validate_arguments
def add_perimeter_from_makesense(project_root_directory: DirectoryPath, make_copy: bool = True):
    perimeter_directory_path = get_perimeter_directory_path(project_root_directory)
    settings = load_settings(project_root_directory)

    perimeter_path = plyer.filechooser.open_file()
    if not perimeter_path:
        return print("Cancelled by user")
    perimeter_path = Path(perimeter_path[0])
    new_perimeter_in_project_path = perimeter_directory_path / perimeter_path.name
    if new_perimeter_in_project_path.exists():
        msg = f"{perimeter_path.name} is already in the project"
        raise ValueError(msg)

    shape, label = get_perimeter_data(perimeter_path)

    settings["perimeters"].append({"label": label, "shape": shape})
    with open(get_project_settings_path(project_root_directory), "wb") as in_yaml:
        yaml.dump(settings, in_yaml)

    if make_copy:
        shutil.copyfile(perimeter_path, perimeter_directory_path / perimeter_path.name)


def get_perimeter_data(perimeter_path: FilePath):
    split_file_stem = perimeter_path.stem.split("-")
    assert split_file_stem[0].lower().endswith("perimeter")
    assert len(split_file_stem) == 3
    # return {"shape": split_file_stem[1], "label": split_file_stem[2]}
    return split_file_stem[1:]
