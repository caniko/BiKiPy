"""
===================
The Sequence Method
===================
Designed for working with sequences of trials.

Rules
=====
- Delimiting is with a dash, "-". Example: 1-Training. Reminder to delimit -> (delimit!)
- The tracking data is segregated into trial-sets. A trial-set consists of a sequence of trial tracking files.
  The trial tracking file has the sequence index stored in as a prefix in the file-stem as a number (delimit!).
  Optionally, for improved readability you can store a sequence label followed by the stage index.
  Example: 0-Habituation.h5, 1-Training.h5, 2-Test.h5.
- Meter pixel ratio must be defined
    - Define it yourself, and plug it into sequence_generate_configuration()
    - If the experiment is in a confined box, or you know the length of a temporally fixed line in your video:
        1. Grab a video frame from one of the trial videos
        2. Define the line in MakeSense
        3. Make "Perimeter" directory in the base folder if it doesn't already exist.
        4. Export as csv and store in the "Perimeter" directory as "meter_pixel_ratio_{meter_length}.csv"; where
           meter_length is the length of the line in meters.

Structure
=========
- Each trial-set is stored in a directory prefixed with the animal ID (delimit!).
- Optional, trial-set metadata; yaml format. Stored inside trial-set directory. Fields in metadata:
    - Optional, perimeter_set. Example: perimeter_set: A
- Project metadata; .xlsx or .odt, xlsx has best support (apologies to FOSS):
  The table must be in the sheet that is on index 0! The metadata file is stored on the root/base folder.
    - Animal ID column name must be "Animal"
    - Genetic state column must have the name "Gene"
    - Optional, "Cohort"
    - Optional, "Sex"
    - Optional, store the usage of a perimeter "Perimeter_{label_of_perimeter}". Row must be empty if there is no
      perimeter. Row must define the label to apply to the perimeter.
    - Make sure your dataset has no junk/invisible characters that might lead to problems with recognizing tags and
      performing comparisons.
- Only MakeSense perimeters are supported. These are stored in the "Perimeter" directory/folder.
    - The file-stem is the perimeter set ID (PID).
      If a perimeter set is stored in several files you must also include a unique identifier (UID)
      after the perimeter set ID (delimit!).
      Opinion: The unique identifier could be a sequence of numbers, letters, or random.
      Example: A-1.csv; where A is the perimeter set ID and 1 is the unique identifier.
    - Make sure that you don't use the same label for the different perimeters when they are defined in MakeSense.
      You can change the label in the file if you have to ensure this later.
    - Optionally, for inspection, you can include an image with the perimeter set as the file-stem.
      Optionally, include the uid if it is specific to the subset (delimit!).
"""
import os
import pickle
from glob import iglob
from logging import getLogger
from pathlib import Path
from typing import Any, Union, Literal

import pandas as pd
import plyer
import yaml
from pydantic import validate_arguments, DirectoryPath, FilePath

from bikipy.behaviour.base import BaseExperiment
from bikipy.perimeter.radial.circle import CirclePerimeter
from bikipy.reader.ingress.cm_pixel_ratio import MeterPixelRatio
from bikipy.utils.io.makesense import from_makesense_coco_polygon, from_makesense_csv_rectangle

logger = getLogger(__name__)


@validate_arguments
def sequence_generate_configuration(
    root_directory: DirectoryPath,
    experiment_class: BaseExperiment,
    meter_pixel_ratio_kwargs: Union[float, dict[str, Any]],
    kinematic_data_file_extension: str = "h5",
    metadata_filename: str = "metadata.xlsx",
    animals_have_several_trial_sets: bool = False,
) -> FilePath:
    meter_pixel_ratio = (
        MeterPixelRatio(**meter_pixel_ratio_kwargs)
        if isinstance(meter_pixel_ratio_kwargs, dict)
        else meter_pixel_ratio_kwargs
    )

    logger.info(f"Generating experiment configuration at {root_directory}")

    animal_ids, trial_set_stage_ids = set(), set()
    for trial_set_dir in os.listdir(root_directory):
        animal_id = trial_set_dir.split("-")[0]
        if animals_have_several_trial_sets and animal_id in animal_ids:
            msg = (
                f"Animal ID {animal_id} is repeated across trial sets. "
                f"Set animals_have_several_trial_sets to true if this behaviour is expected"
            )
            raise ValueError(msg)
        animal_ids.add(animal_id)

        stage_ids = set()
        for filename in iglob(str(Path(trial_set_dir) / f"*.{kinematic_data_file_extension}")):
            stage_ids.add(filename.split("-")[0])
        trial_set_stage_ids.add(stage_ids)

    if len(trial_set_stage_ids) != 1:
        msg = f"The trial sets do not have identical trial stage sequence:\n{trial_set_stage_ids}"
        raise ValueError(msg)

    stages = trial_set_stage_ids.pop()
    number_of_stages = len(stages)
    if experiment_class.stage_index_to_trial_class and number_of_stages != len(
        experiment_class.stage_index_to_trial_class
    ):
        msg = (
            f"The trials stage length are incorrect, {number_of_stages}."
            f"experiment_class.stage_index_to_trial_class:\n{experiment_class.stage_index_to_trial_class}"
        )
        raise ValueError(msg)

    with open(metadata_filename, "rb") as in_file:
        metadata_df = pd.read_excel(in_file)

    try:
        metadata_animal_id_column_set = set(metadata_df.loc["Animal"])
    except KeyError:
        msg = f"Animal ID column, Animal, is not defined in the metadata sheet. Defined columns:\n{metadata_df.columns}"
        raise KeyError(msg)

    if metadata_animal_id_column_set != animal_ids:
        msg = (
            "The animal ID sets in the metadata and the trial_set directory names do not match:\n"
            f"- metadata: {metadata_animal_id_column_set}\n- trial_sets: {animal_ids}"
        )
        raise ValueError(msg)

    settings_path = _get_project_settings_path(root_directory)
    with open(settings_path, "w") as out_file:
        yaml.dump(
            {
                "meter_pixel_ratio": meter_pixel_ratio,
                "metadata_filename": metadata_filename,
                "required_fields": dict.fromkeys(experiment_class.schema()["required"]),
                "perimeter": {
                    "perimeter_pickle_file": "perimeters.pickle",
                    "immutable": {
                        "perimeters_added": False
                    }
                },
                "immutable": {
                    "kinematic_data_file_extension": kinematic_data_file_extension,
                    "experiment_class": experiment_class.__name__,
                    "Total # animals": len(animal_ids),
                    "animals_have_several_trial_sets": animals_have_several_trial_sets,
                },
            },
            out_file
        )

    return settings_path


@validate_arguments
def add_perimeter_from_makesense(
    root_directory: DirectoryPath, shape: Literal["circle", "rectangle", "polygon", "parallelogram"]
):
    perimeter_pickle_path = _get_perimeter_pickle_path(root_directory, _load_settings(root_directory))
    if perimeter_pickle_path.exists():
        with open(perimeter_pickle_path, "rb") as in_file:
            perimeters = pickle.load(in_file)
    else:
        perimeters = {}

    perimeter_path = plyer.filechooser.open_file()
    if not perimeter_path:
        return print("Cancelled by user")
    perimeter_path = Path(perimeter_path[0])

    if shape == "circle":
        perimeters.update(CirclePerimeter.from_makesense_line(perimeter_path))
    elif shape == "rectangle":
        perimeters.update(from_makesense_csv_rectangle(perimeter_path))
    elif shape == "polygon" or shape == "parallelogram":
        perimeters.update(from_makesense_coco_polygon(perimeter_path, map_to_label=True))
    else:
        raise RuntimeError

    with open(perimeter_pickle_path, "wb") as out_file:
        pickle.dump(perimeters, out_file)


@validate_arguments
def sequence_ingress_method(root_directory: DirectoryPath):
    pass


def _get_project_settings_path(root_directory: DirectoryPath):
    return root_directory / "settings.yaml"


def _load_settings(root_directory: DirectoryPath):
    with open(_get_project_settings_path(root_directory), "r") as in_file:
        return yaml.load(in_file, yaml.full_load)


def _get_perimeter_pickle_path(root_directory: DirectoryPath, settings: dict):
    return root_directory / settings["perimeter_pickle_file"]
