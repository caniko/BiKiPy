import pickle
from pathlib import Path
from typing import Literal

import plyer
import yaml
from pydantic import DirectoryPath, validate_arguments, FilePath

from bikipy.perimeter.radial.circle import CirclePerimeter
from bikipy.reader.ingress.utils.constant import get_perimeter_pickle_path, get_perimeter_dir_path, load_settings
from bikipy.utils.io.makesense import from_makesense_csv_rectangle, from_makesense_coco_polygon


SHAPE_TYPING = Literal['circle', 'parallelogram', 'polygon', 'rectangle']
ALL_SHAPES = {"circle", "polygon", "parallelogram", "rectangle"}


@validate_arguments
def init_all_perimeters(root_directory: DirectoryPath, shape: SHAPE_TYPING = None) -> tuple:
    if not (perimeter_dir := get_perimeter_dir_path(root_directory)).exists():
        return ()
    perimeters = {
        filename.stem.split("-")[1]: _create_perimeter_object(perimeter_dir / filename, shape)
        for filename in perimeter_dir.glob(r"(P|p)erimeter-*")
    }
    with open(get_perimeter_pickle_path(perimeter_dir, load_settings(root_directory)), "wb") as out_file:
        pickle.dump(perimeters, out_file)
    return tuple(perimeters)


@validate_arguments
def add_perimeter_from_makesense(
    root_directory: DirectoryPath, make_copy: bool = True
):
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

    perimeters.update(_create_perimeter_object(perimeter_path))

    with open(perimeter_pickle_path, "wb") as out_file:
        pickle.dump(perimeters, out_file)

    with open


def _create_perimeter_object(perimeter_path: FilePath):
    match perimeter_path.stem.split("-")[1].strip().lower():
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
