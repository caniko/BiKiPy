import shutil
from logging import getLogger
from pathlib import Path
from typing import Literal, Optional, Mapping

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

logger = getLogger(__name__)


@validate_arguments
def detect_perimeters_in_project(root_directory: DirectoryPath, create_object: bool = False) -> list:
    perimeter_dir = get_perimeter_dir_path(root_directory)
    detection_data = []
    for filename in perimeter_dir.glob("perimeter-*"):
        perimeter_path = root_directory / filename
        shape, label = get_perimeter_data(perimeter_path)
        data = {"label": label, "shape": shape}
        if create_object:
            data["perimeter"] = create_perimeter_object(perimeter_path, shape)
        detection_data.append(data)
    if not detection_data:
        msg = (
            "No perimeter data was found. Set perimeter strategy to None or "
            'revise perimeter filenames to the correct format, "perimeter-{label}"'
        )
        raise ValueError(msg)
    return detection_data


def generate_label_to_object_field(root_directory: DirectoryPath):
    return {
        perimeter_data.pop("label"): perimeter_data
        for perimeter_data in detect_perimeters_in_project(root_directory, create_object=True)
    }


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

    shape, label = get_perimeter_data(perimeter_path)

    settings["perimeters"].append({"label": label, "shape": shape})
    with open(get_project_settings_path(root_directory), "wb") as in_yaml:
        yaml.dump(settings, in_yaml)

    if make_copy:
        shutil.copyfile(perimeter_path, perimeter_dir_path / perimeter_path.name)


@validate_arguments
def create_perimeter_object(perimeter_path: FilePath, shape: Optional[Literal["circle", "parallelogram", "polygon", "rectangle"]] = None):
    match shape or get_perimeter_data(perimeter_path)[0]:
        case "circle":
            return CirclePerimeter.from_makesense_line(perimeter_path)
        case "rectangle":
            return from_makesense_csv_rectangle(perimeter_path)
        case "polygon" | "parallelogram":
            return from_makesense_coco_polygon(perimeter_path, map_to_label=True)
        case _:
            raise ValueError


def get_trial_perimeter_label_from_metadata(animal_id_row: Mapping, settings: dict, stage: Optional[int] = None) -> str:
    return animal_id_row["Perimeter"][stage] if settings["stageful_metadata"] and stage else animal_id_row["Perimeter"]


def get_perimeter_data(perimeter_path: FilePath):
    split_file_stem = perimeter_path.stem.split("-")
    assert split_file_stem[0].lower() == "perimeter"
    assert len(split_file_stem) == 3
    # return {"shape": split_file_stem[1], "label": split_file_stem[2]}
    return split_file_stem[1:]
