import json
import os
from functools import reduce
from logging import getLogger
from pathlib import Path

import pandas as pd
import yaml
from pydantic import DirectoryPath, validate_arguments

from bikipy.behaviour.mapping import EXPERIMENT_NAME_TO_CLASS
from bikipy.ingress.plugin.center import detect_center_in_perimeter_directory
from bikipy.ingress.utils.io import (
    get_dataset_dir_path,
    get_perimeter_dir_path,
    get_project_settings_path,
    initialize_metadata_data_frame,
    load_settings,
)
from bikipy.ingress.utils.perimeter import (
    generate_label_to_object_field,
    get_trial_perimeter_label_from_metadata,
    partial_first_perimeter_set_from_makesense_from_settings,
)

logger = getLogger(__name__)


@validate_arguments
def sequence_generate_configuration(
    project_root_dir: DirectoryPath,
    experiment_name: str,
    kinematic_data_file_extension: str = ".h5",
    dry_run: bool = False,
) -> dict:
    from bikipy.ingress.core import init_settings

    experiment_class = EXPERIMENT_NAME_TO_CLASS[experiment_name.strip().lower()]

    logger.info(f"Generating experiment configuration at {project_root_dir}")

    method_settings = {
        "ingress_method": "sequence",
        "animal-ID_absent_from_metadata": "ignore",  # TODO
        "sequence_index_to_trial_class_name": {
            i: trial_class_name for i, trial_class_name in enumerate(experiment_class.trial_class_names)
        },
    }
    method_immutable = {
        "detected_animal_ids": _animal_ids,
    }

    settings = init_settings(
        experiment_class, method_settings, project_root_dir, kinematic_data_file_extension, method_immutable
    )

    if dry_run:
        print(json.dumps(settings, indent=2))
    else:
        settings_path = get_project_settings_path(project_root_dir)
        with open(settings_path, "w") as out_file:
            yaml.safe_dump(settings, out_file, sort_keys=False)

    return settings


@validate_arguments
def sequence_analysis_keyword_arguments(
    project_root_dir: DirectoryPath, metadata_plugin_name_to_label_to_parameter: dict
):
    def get_plugin_index_from_stageful_metadata(feature_sheet_header: str):
        return animal_metadata.loc[:, [feature_sheet_header, sequence_index]][0]

    def get_plugin_index_from_

    settings = load_settings(project_root_dir)
    metadata = initialize_metadata_data_frame(project_root_dir, settings["ingress"]["stageful_metadata"])

    partial_first_perimeter_set_from_makesense = partial_first_perimeter_set_from_makesense_from_settings(settings)

    dataset_directory_path = _dataset_directory_path(project_root_dir)

    trial_id_to_trial_class_name, trial_id_to_keyword_arguments, metadata_index_to_trial_id = {}, {}, {}
    for animal_id in os.listdir(dataset_directory_path):
        animal_id = int(animal_id)
        animal_dir = dataset_directory_path / str(animal_id)

        animal_metadata = metadata.loc[animal_id, :]

        trial_ids = []
        for trial_data_filename in animal_dir.glob(f"*{settings['immutable']['kinematic_data_file_extension']}"):
            trial_data_filename = Path(trial_data_filename)
            sequence_index = int(trial_data_filename.stem.split(".")[0])

            trial_id = _define_trial_id(animal_id, sequence_index)
            trial_ids.append(trial_id)

            trial_id_to_trial_class_name[trial_id] = settings["sequence_index_to_trial_class_name"][sequence_index]
            trial_id_to_keyword_arguments[trial_id] = {
                "label": trial_id,
                "animal_id": animal_id,
                "stage": sequence_index,
                "coordinate_data_path": trial_data_filename,
            }

            if settings["ingress"]["meters_per_pixel_definition_strategy"]:
                if "meters_per_pixel" in metadata_plugin_name_to_label_to_parameter:
                    trial_id_to_keyword_arguments[trial_id][
                        "rectangle_meters_per_pixel_point"
                    ] = metadata_plugin_name_to_label_to_parameter["meters_per_pixel"][
                        get_plugin_index_from_stageful_metadata("Meters per pixel")
                    ]
            if settings["ingress"]["perimeter_definition_strategy"]:
                if "perimeter" in metadata_plugin_name_to_label_to_parameter:
                    trial_id_to_keyword_arguments[trial_id][
                        "rectangle_center_point"
                    ] = metadata_plugin_name_to_label_to_parameter["perimeter"][
                        get_plugin_index_from_stageful_metadata("Perimeter")
                    ]

                elif settings["ingress"]["perimeter_definition_strategy"] == "trialwise":
                    perimeter_sets = []
                    for perimeter_path in animal_dir.glob(f"{sequence_index}.perimeter*"):
                        perimeter_sets.append(partial_first_perimeter_set_from_makesense(perimeter_path))

                    if not (length := len(perimeter_sets)):
                        logger.debug(f"No perimeters were found for Animal #{animal_id} for sequence {sequence_index}")
                    else:
                        perimeter_set = reduce(lambda a, b: a + b, perimeter_sets) if length != 1 else perimeter_sets[0]

                else:
                    msg = (
                        f"{settings['perimeter']['perimeter_definition_strategy']} as perimeter definition strategy "
                        f"is not supported with the {settings['ingress_method']} ingress method"
                    )
                    raise ValueError(msg)

                trial_id_to_keyword_arguments[trial_id].update(perimeter_set.label_to_perimeter)
            if settings["ingress"]["center_definition_strategy"]:
                if "center" in metadata_plugin_name_to_label_to_parameter:
                    trial_id_to_keyword_arguments[trial_id][
                        "rectangle_center_point"
                    ] = metadata_plugin_name_to_label_to_parameter["center"][
                        get_plugin_index_from_stageful_metadata("Center")
                    ]

        metadata_index_to_trial_id[animal_id] = tuple(trial_ids)

    return {
        "trial_id_to_trial_class_name": trial_id_to_trial_class_name,
        "trial_id_to_keyword_arguments": trial_id_to_keyword_arguments,
    }, metadata_index_to_trial_id


def verify_project_structure():
    animal_ids = set()
    trial_set_stage_ids = []
    for trial_set_dir in get_dataset_dir_path(project_root_dir).iterdir():
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

    metadata_df = initialize_metadata_data_frame(project_root_dir, True)

    try:
        metadata_animal_id_column_set = set(metadata_df.index)
    except KeyError:
        msg = f"Animal ID column, Animal, is not defined in the metadata sheet. Defined columns:\n{metadata_df.columns}"
        raise KeyError(msg)

    if metadata_animal_id_column_set.issubset(animal_ids):
        msg = (
            "The animal ID sets in the metadata and the trial_set directory names do not match:\n"
            f"- metadata: {sorted(metadata_animal_id_column_set)}\n- trial_sets: {sorted(animal_ids)}"
        )
        raise ValueError(msg)


def _animal_ids(project_root_dir: DirectoryPath):
    animal_ids = set()
    for trial_set_dir in get_dataset_dir_path(project_root_dir).iterdir():
        if trial_set_dir.is_dir():
            animal_ids.add(trial_set_dir.name.split("-")[0])
    return animal_ids


def _dataset_directory_path(project_root_dir: DirectoryPath):
    return project_root_dir / "dataset"


def _define_trial_id(animal_id: int | str, sequence_index: int | str):
    return f"{animal_id}-{sequence_index}"
