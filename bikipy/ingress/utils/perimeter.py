import pickle
import shutil
from pathlib import Path
from typing import Literal

import plyer
import yaml
from pydantic import DirectoryPath, FilePath, validate_arguments

from bikipy.ingress.utils.constant import (
    get_perimeter_dir_path,
    get_perimeter_pickle_path,
    get_project_settings_path,
    load_settings,
)
from bikipy.perimeter.radial.circle import CirclePerimeter
from bikipy.utils.io.makesense import (
    from_makesense_coco_polygon,
    from_makesense_csv_rectangle,
)

SHAPE_TYPING = Literal["circle", "parallelogram", "polygon", "rectangle"]
ALL_SHAPES = {"circle", "polygon", "parallelogram", "rectangle"}
PERIMETER_PICKLE_FILE_NAME = "perimeters.pickle"


@validate_arguments
def init_all_perimeters(root_directory: DirectoryPath, dry_run: bool = False) -> dict:
    perimeter_dir = get_perimeter_dir_path(root_directory)
    if not perimeter_dir.exists() and not any(perimeter_dir.glob(r"(perimeter-*")):
        return {"perimeter": {"perimeter_pickle_file": PERIMETER_PICKLE_FILE_NAME, "info": {}}}
    perimeters, info = {}, {}
    for filename in perimeter_dir.glob("perimeter-*"):
        shape, label = _get_perimeter_data(root_directory / filename)
        perimeters[label] = _create_perimeter_object(perimeter_dir / filename, shape)
        info[label] = shape

    if not dry_run:
        with open(perimeter_dir / PERIMETER_PICKLE_FILE_NAME, "wb") as out_file:
            pickle.dump(perimeters, out_file)
    return {"perimeter": {"perimeter_pickle_file": PERIMETER_PICKLE_FILE_NAME, "info": info}}


@validate_arguments
def add_perimeter_from_makesense(root_directory: DirectoryPath, make_copy: bool = True):
    perimeter_dir_path = get_perimeter_dir_path(root_directory)
    settings = load_settings(root_directory)
    perimeter_pickle_path = get_perimeter_pickle_path(perimeter_dir_path, settings)

    if perimeter_pickle_path.exists():
        with open(perimeter_pickle_path, "rb") as in_file:
            perimeters = pickle.load(in_file)
    else:
        perimeters = {}

    perimeter_path = plyer.filechooser.open_file()
    if not perimeter_path:
        return print("Cancelled by user")
    perimeter_path = Path(perimeter_path[0])
    new_perimeter_in_project_path = perimeter_dir_path / perimeter_path.name
    if new_perimeter_in_project_path.exists():
        msg = f"{perimeter_path.name} is already in the project"
        raise ValueError(msg)

    shape, label = _get_perimeter_data(perimeter_path)
    perimeters[label] = _create_perimeter_object(perimeter_path, shape)

    with open(perimeter_pickle_path, "wb") as out_file:
        pickle.dump(perimeters, out_file)

    settings["perimeters"]["info"][label] = shape
    with open(get_project_settings_path(root_directory), "wb") as in_yaml:
        yaml.dump(settings, in_yaml)

    if make_copy:
        shutil.copyfile(perimeter_path, perimeter_dir_path / perimeter_path.name)


@validate_arguments
def _create_perimeter_object(perimeter_path: FilePath, shape: SHAPE_TYPING):
    match shape:
        case "circle":
            return CirclePerimeter.from_makesense_line(perimeter_path)
        case "rectangle":
            return from_makesense_csv_rectangle(perimeter_path)
        case "polygon" | "parallelogram":
            return from_makesense_coco_polygon(perimeter_path, map_to_label=True)
        case _:
            raise ValueError


def _get_perimeter_data(perimeter_path: FilePath):
    split_file_stem = perimeter_path.stem.split("-")
    assert split_file_stem[0].lower() == "perimeter"
    assert len(split_file_stem) == 3
    # return {"shape": split_file_stem[1], "label": split_file_stem[2]}
    return split_file_stem[1:]
