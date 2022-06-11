import json
from functools import cached_property, reduce
from logging import getLogger
from pathlib import Path

import yaml
from pydantic import DirectoryPath, validate_arguments

from bikipy.behaviour.mapping import EXPERIMENT_NAME_TO_CLASS
from bikipy.ingress.core import BaseIngress
from bikipy.ingress.plugin import PLUGIN_NAME_TO_KEYRING
from bikipy.ingress.utils.io import (
    get_dataset_directory_path,
    get_project_settings_path,
)

logger = getLogger(__name__)


class SequenceIngress(BaseIngress):
    @cached_property
    def _experiment_class_kwargs_metadata_index_to_trial_id_getter(self):
        def get_plugin_index_from_stageful_metadata(feature_sheet_header: str):
            feature_column = animal_metadata[feature_sheet_header]

            if not self.stageful_metadata:
                return feature_column

            if len(feature_column) == 1:
                return feature_column[0]
            return feature_column[sequence_index]

        trial_id_to_trial_class_name, trial_id_to_keyword_arguments, metadata_index_to_trial_id = {}, {}, {}
        for animal_dir in self.dataset_directory_path.iterdir():
            animal_id = int(animal_dir.stem)

            try:
                animal_metadata = self.metadata.loc[animal_id, :]
            except KeyError:
                logger.debug(f"Animal ID {animal_id} is absent from the metadata index, skipping the trial-set")
                continue

            trial_ids = []
            for trial_data_filename in animal_dir.glob(f"*{self.kinematic_data_file_extension}"):
                trial_data_filename = Path(trial_data_filename)
                sequence_index = int(trial_data_filename.stem.split(".")[0])

                trial_id = _define_trial_id(animal_id, sequence_index)
                trial_ids.append(trial_id)

                trial_id_to_trial_class_name[trial_id] = self.settings["sequence_index_to_trial_class_name"][
                    sequence_index
                ]
                trial_id_to_keyword_arguments[trial_id] = {
                    # "label": trial_id,    Already in BaseExperiment
                    "animal_id": animal_id,
                    "stage": sequence_index,
                    "coordinate_data_path": trial_data_filename,
                }

                for plugin_name, keyring in PLUGIN_NAME_TO_KEYRING.items():
                    if self.settings["ingress"][keyring["ingress_key"]] != "metadata":
                        continue

                    trial_id_to_keyword_arguments[trial_id][keyring["bikipy_trial_key"]] = self.get_plugin_parameter(
                        keyring["code_key"], get_plugin_index_from_stageful_metadata
                    )

                if self.settings["ingress"]["perimeter_definition_strategy"] == "trialwise":
                    perimeter_sets = []
                    for perimeter_path in animal_dir.glob(f"{sequence_index}.perimeter*"):
                        perimeter_sets.append(
                            self.partial_first_perimeter_set_from_makesense_from_settings(perimeter_path)
                        )

                    if length := len(perimeter_sets):
                        perimeter_set = reduce(lambda a, b: a + b, perimeter_sets) if length != 1 else perimeter_sets[0]
                    else:
                        msg = f"No perimeters were found for Animal #{animal_id} for sequence {sequence_index}"
                        raise ValueError(msg)

                    trial_id_to_keyword_arguments[trial_id].update(perimeter_set.label_to_perimeter)

            metadata_index_to_trial_id[animal_id] = tuple(trial_ids)

        return {
            "trial_id_to_trial_class_name": trial_id_to_trial_class_name,
            "trial_id_to_keyword_arguments": trial_id_to_keyword_arguments,
        }, metadata_index_to_trial_id

    def verify_project_structure(self):
        animal_ids = set()
        trial_set_stage_ids = []
        for trial_set_dir in self.dataset_directory_path.iterdir():
            if trial_set_dir.is_file() or trial_set_dir.name == "perimeter":
                continue

            animal_id = trial_set_dir.name.split("-")[0]
            animal_ids.add(animal_id)

            stage_ids = set()
            for filename in trial_set_dir.glob(f"*{self.kinematic_data_file_extension}"):
                stage_ids.add(int(filename.stem.split("-")[0]))
            trial_set_stage_ids.append(stage_ids)

        if any(trial_set_stage_ids[0] != stage_ids for stage_ids in trial_set_stage_ids[1:]):
            msg = f"The trial sets do not have identical trial stage sequence:\n{trial_set_stage_ids}"
            raise ValueError(msg)

        try:
            metadata_animal_id_column_set = set(self.metadata.index)
        except KeyError:
            msg = f"Animal ID column, Animal, is not defined in the metadata sheet. Defined columns:\n{self.metadata.columns}"
            raise KeyError(msg)

        if metadata_animal_id_column_set.issubset(animal_ids):
            msg = (
                "The animal ID sets in the metadata and the trial_set directory names do not match:\n"
                f"- metadata: {sorted(metadata_animal_id_column_set)}\n- trial_sets: {sorted(animal_ids)}"
            )
            raise ValueError(msg)


@validate_arguments
def sequence_generate_configuration(
    project_root_directory: DirectoryPath,
    experiment_name: str,
    kinematic_data_file_extension: str = ".h5",
    dry_run: bool = False,
) -> dict:
    from bikipy.ingress.core import init_settings

    experiment_class = EXPERIMENT_NAME_TO_CLASS[experiment_name.strip().lower()]

    logger.info(f"Generating experiment configuration at {project_root_directory}")

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
        experiment_class, method_settings, project_root_directory, kinematic_data_file_extension, method_immutable
    )

    if dry_run:
        print(json.dumps(settings, indent=2))
    else:
        settings_path = get_project_settings_path(project_root_directory)
        with open(settings_path, "w") as out_file:
            yaml.safe_dump(settings, out_file, sort_keys=False)

    return settings


def _animal_ids(project_root_directory: DirectoryPath):
    animal_ids = set()
    for trial_set_dir in get_dataset_directory_path(project_root_directory).iterdir():
        if trial_set_dir.is_dir():
            animal_ids.add(trial_set_dir.name.split("-")[0])
    return animal_ids


def _define_trial_id(animal_id: int | str, sequence_index: int | str):
    return f"{animal_id}_{sequence_index}"
