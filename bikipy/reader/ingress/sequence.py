"""
===================
The Sequence Method
===================
Designed for working with sequences of trials.

All delimiting is with a dash, "-"

The sequence method:
    - Each trial set has its own directory, the name of the directory must be prefixed with the animal ID (delimit!).
    - Dataset of each component of the trial has the stage index as prefix, stage indexing starts from 0 (delimit!).
      Optionally, for improved readability one can have the stage index followed by the stage label.
      Example: 0-Habituation, 1-Training, 2-Test.
    - The metadata must be either .xlsx or .odt (xlsx has best support, sorry FOSS), the metadata must be in sheet 0!
        - Animal ID column name must be "Animal"
        - Genetic state column must have the name "Gene"
        - Optional, "Cohort"
        - Optional, "Sex"
        - Optional, Store the usage of a perimeter "Perimeter_{label_of_perimeter}". Row must be empty if the
          perimeter. Row must define the label to apply to the perimeter
        - Make sure your dataset has no junk characters that might lead to problems with string comparisons
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
from bikipy.reader.ingress.cm_pixel_ratio import CentimeterPixelRatio
from bikipy.utils.io.makesense import from_makesense_coco_polygon

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
        CentimeterPixelRatio(**meter_pixel_ratio_kwargs)
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
    root_directory: DirectoryPath, shape: Literal["circle", "polygon", "parallelogram"]
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
    elif shape == "polygon" or shape == "parallelogram":
        perimeters.update(from_makesense_coco_polygon(perimeter_path, map_to_label=True))
    else:
        raise RuntimeError

    with open(perimeter_pickle_path, "wb") as out_file:
        pickle.dump(perimeters, out_file)


@validate_arguments
def sequence_ingress_method(root_directory: DirectoryPath):



def _get_project_settings_path(root_directory: DirectoryPath):
    return root_directory / "settings.yaml"


def _load_settings(root_directory: DirectoryPath):
    with open(_get_project_settings_path(root_directory), "r") as in_file:
        return yaml.load(in_file, yaml.full_load)


def _get_perimeter_pickle_path(root_directory: DirectoryPath, settings: dict):
    return root_directory / settings["perimeter_pickle_file"]
