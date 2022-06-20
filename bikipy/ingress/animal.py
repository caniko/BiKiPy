from logging import getLogger
from pathlib import Path
from typing import ClassVar

from pydantic import DirectoryPath, validate_arguments, FilePath

from bikipy.behaviour.mapping import EXPERIMENT_NAME_TO_CLASS
from bikipy.ingress.core import BaseIngress
from bikipy.ingress.utils.io import get_dataset_directory_path

logger = getLogger(__name__)


class AnimalIngress(BaseIngress):
    ingress_method: ClassVar[str] = "animal"

    def _ingress_reader(self):
        def sequence_index_from_file_path(file_path: FilePath) -> int:
            return int(file_path.stem.split(".")[0])

        for animal_dir in self.dataset_directory_path.iterdir():
            animal_id = int(animal_dir.stem) if animal_dir.stem.isdigit() else animal_dir.stem

            for trial_data_filename in animal_dir.glob(f"*{self.kinematic_data_file_extension}"):
                trial_data_filename = Path(trial_data_filename)
                sequence_index = sequence_index_from_file_path(trial_data_filename)

                trial_id = _define_trial_id(animal_id, sequence_index)
                trial_id = int(trial_id) if trial_id.isdigit() else trial_id

                self._trial_id_to_trial_class_name[trial_id] = self.experiment_class.stage_index_to_trial_class_name[
                    sequence_index
                ]
                self._trial_id_to_keyword_arguments[trial_id] = {
                    "label": trial_id,
                    "animal_id": animal_id,
                    "stage": sequence_index,
                    "coordinate_data_path": trial_data_filename,
                }
                for plugin_info in self._trial_wise_plugins:
                    for plugin_data_file in animal_dir.glob(f"*{plugin_info['code_key']}*"):
                        self._trial_id_to_keyword_arguments[trial_id][plugin_info["bikipy_trial_key"]] = plugin_info[
                            "file_path_to_value"
                        ](plugin_data_file, trial_id, self)

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

    @property
    def method_settings(self):
        return sequence_generate_configuration(
            self.project_root_directory, self.experiment_name, self.kinematic_data_file_extension
        )


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
        "animal-ID_absent_from_metadata": "ignore",  # TODO
        "sequence_index_to_trial_class_name": {
            sequence_index: trial_class_name
            for sequence_index, trial_class_name in enumerate(experiment_class.trial_class_names)
        },
    }
    method_immutable = {
        "detected_animal_ids": list(_animal_ids(project_root_directory)),
    }

    return init_settings(
        "animal",
        project_root_directory,
        experiment_class,
        method_settings,
        kinematic_data_file_extension,
        method_immutable,
        dry_run,
    )


def _animal_ids(project_root_directory: DirectoryPath):
    animal_ids = set()
    for trial_set_dir in get_dataset_directory_path(project_root_directory).iterdir():
        if trial_set_dir.is_dir():
            animal_ids.add(trial_set_dir.name.split("-")[0])
    return animal_ids


def _define_trial_id(animal_id: int | str, sequence_index: int | str):
    return f"{animal_id}_{sequence_index}"
