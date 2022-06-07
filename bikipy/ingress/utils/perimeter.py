import shutil
from logging import getLogger
from pathlib import Path
from typing import Literal, Optional

import pandas as pd
import plyer
import yaml
from pydantic import DirectoryPath, FilePath, validate_arguments

from bikipy.ingress.utils.constant import (
    get_perimeter_dir_path,
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


logger = getLogger(__name__)


@validate_arguments
def detect_perimeters_in_project(root_directory: DirectoryPath) -> list:
    perimeter_dir = get_perimeter_dir_path(root_directory)
    detection_data = []
    for filename in perimeter_dir.glob("perimeter-*"):
        shape, label = _get_perimeter_data(root_directory / filename)
        detection_data.append({"label": label, "shape": shape})
    if not detection_data:
        logger.info("No perimeter data was found. Ignore if no perimeters are required for analysis.")
    return detection_data


@validate_arguments
def refresh_perimeters_in_project(root_directory: DirectoryPath) -> None:
    settings = load_settings(root_directory)
    settings["perimeters"] = detect_perimeters_in_project(root_directory)
    with open(get_project_settings_path(root_directory), "wb") as in_yaml:
        yaml.dump(settings, in_yaml)


@validate_arguments
def add_perimeter_from_makesense(root_directory: DirectoryPath, make_copy: bool = True):
    perimeter_dir_path = get_perimeter_dir_path(root_directory)
    settings = load_settings(root_directory)

    perimeter_path = plyer.filechooser.open_file()
    if not perimeter_path:
        return print("Cancelled by user")
    perimeter_path = Path(perimeter_path[0])
    new_perimeter_in_project_path = perimeter_dir_path / perimeter_path.name
    if new_perimeter_in_project_path.exists():
        msg = f"{perimeter_path.name} is already in the project"
        raise ValueError(msg)

    shape, label = _get_perimeter_data(perimeter_path)

    settings["perimeters"].append({"label": label, "shape": shape})
    with open(get_project_settings_path(root_directory), "wb") as in_yaml:
        yaml.dump(settings, in_yaml)

    if make_copy:
        shutil.copyfile(perimeter_path, perimeter_dir_path / perimeter_path.name)


@validate_arguments
def create_perimeter_object(perimeter_path: FilePath, shape: SHAPE_TYPING):
    match shape:
        case "circle":
            return CirclePerimeter.from_makesense_line(perimeter_path)
        case "rectangle":
            return from_makesense_csv_rectangle(perimeter_path)
        case "polygon" | "parallelogram":
            return from_makesense_coco_polygon(perimeter_path, map_to_label=True)
        case _:
            raise ValueError


def get_perimeter_objects(root_directory: DirectoryPath) -> list:
    perimeter_dir = detect_perimeters_in_project(root_directory)
    detection_data = []
    for filename in perimeter_dir.glob("perimeter-*"):
        shape, label = _get_perimeter_data(root_directory / filename)
        detection_data.append({"label": label, "shape": shape})
    if not detection_data:
        logger.info("No perimeter data was found. Ignore if no perimeters are required for analysis.")
    return detection_data


def get_trial_perimeter_label_from_metadata(animal_id_row: pd.DataFrame, stage: Optional[int] = None) -> str:
    return animal_id_row["Perimeter"][stage] if stage else animal_id_row["Perimeter"]


def _get_perimeter_data(perimeter_path: FilePath):
    split_file_stem = perimeter_path.stem.split("-")
    assert split_file_stem[0].lower() == "perimeter"
    assert len(split_file_stem) == 3
    # return {"shape": split_file_stem[1], "label": split_file_stem[2]}
    return split_file_stem[1:]
