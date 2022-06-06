import json
import os
import pickle
from functools import lru_cache
from glob import iglob
from itertools import count
from logging import getLogger
from typing import Optional

import pandas as pd
import yaml
from pydantic import DirectoryPath, validate_arguments

from bikipy.behaviour.mapping import NAME_TO_CLASS
from bikipy.ingress.core import init_settings
from bikipy.ingress.utils.constant import get_project_settings_path, load_settings
from bikipy.ingress.utils.meter_pixel_ratio import get_meter_pixel_ratio

logger = getLogger(__name__)


@validate_arguments
def sequence_generate_configuration(
    root_directory: DirectoryPath,
    experiment_name: str,
    meter_pixel_ratio: Optional[float] = None,
    kinematic_data_file_extension: str = ".h5",
    animals_have_plural_trial_sets: bool = False,
    dry_run: bool = False,
) -> dict:
    experiment_class = NAME_TO_CLASS[experiment_name.strip().lower()]

    meter_pixel_ratio = meter_pixel_ratio or get_meter_pixel_ratio(root_directory)

    logger.info(f"Generating experiment configuration at {root_directory}")

    animal_ids = set()
    trial_set_stage_ids = []
    for trial_set_dir in root_directory.iterdir():
        if trial_set_dir.is_file() or trial_set_dir.name == "perimeter":
            continue
        animal_id = trial_set_dir.name.split("-")[0]
        if animals_have_plural_trial_sets and animal_id in animal_ids:
            msg = (
                f"Animal ID {animal_id} is repeated across trial sets. "
                f"Set animals_have_plural_trial_sets to true if this behaviour is expected"
            )
            raise ValueError(msg)
        animal_ids.add(animal_id)

        stage_ids = set()
        for filename in trial_set_dir.glob(f"*{kinematic_data_file_extension}"):
            stage_ids.add(int(filename.stem.split("-")[0]))
        trial_set_stage_ids.append(stage_ids)

    if any(trial_set_stage_ids[0] != stage_ids for stage_ids in trial_set_stage_ids[1:]):
        msg = f"The trial sets do not have identical trial stage sequence:\n{trial_set_stage_ids}"
        raise ValueError(msg)

    with open(root_directory / "metadata.xlsx", "rb") as in_file:
        metadata_df = pd.read_excel(in_file)

    try:
        metadata_animal_id_column_set = set(metadata_df.loc[:, "Animal"])
    except KeyError:
        msg = f"Animal ID column, Animal, is not defined in the metadata sheet. Defined columns:\n{metadata_df.columns}"
        raise KeyError(msg)

    if metadata_animal_id_column_set != animal_ids:
        msg = (
            "The animal ID sets in the metadata and the trial_set directory names do not match:\n"
            f"- metadata: {metadata_animal_id_column_set}\n- trial_sets: {animal_ids}"
        )
        raise ValueError(msg)

    method_settings = {
        "sequence_index_to_trial_class": {i: trial_class for i, trial_class in enumerate(experiment_class.trial_classes)}
    }

    settings = init_settings(
        experiment_class,
        method_settings,
        root_directory,
        meter_pixel_ratio,
        kinematic_data_file_extension,
        animal_ids,
        animals_have_plural_trial_sets,
    )

    if dry_run:
        print(json.dumps(settings, indent=2))
    else:
        settings_path = get_project_settings_path(root_directory)
        with open(settings_path, "w") as out_file:
            yaml.dump(settings, out_file, sort_keys=False)

    return settings


@validate_arguments
def analyse_sequence(root_directory: DirectoryPath):
    settings = load_settings(root_directory)
    dataset_directory_path = _dataset_directory_path(root_directory)
    experiment_class = NAME_TO_CLASS[settings["immutable"]["experiment_class"]]
    metadata = pd.read_excel(root_directory / settings["immutable"]["metadata_filename"])
    metadata.set_index("Animal", inplace=True)

    trial_id_vs_trial_class, trial_id_vs_keyword_arguments = {}, {}
    trial_id_counter = count(start=1)

    for animal_id in os.listdir(dataset_directory_path):
        animal_id = str(animal_id)
        animal_metadata = dict(metadata.loc[animal_id, :])
        for trial_data_filename in iglob(
            str(dataset_directory_path / animal_id / f"*{settings['immutable']['kinematic_data_file_extension']}")
        ):
            trial_id = next(trial_id_counter)
            sequence_index = trial_data_filename.split("-")[0]
            trial_class = settings["sequence_index_to_trial_class"][sequence_index]

            trial_id_vs_keyword_arguments[trial_id] = {
                "animal_id": animal_id,
            }
            trial_id_vs_trial_class[trial_id] = settings["sequence_index_to_trial_class"][sequence_index]


def _dataset_directory_path(root_directory: DirectoryPath):
    return root_directory / "dataset"
