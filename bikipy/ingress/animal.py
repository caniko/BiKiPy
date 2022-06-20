from logging import getLogger
from pathlib import Path
from typing import ClassVar

from pydantic import DirectoryPath, FilePath

from bikipy.ingress.core import BaseIngress
from bikipy.ingress.utils.io import get_dataset_directory_path

logger = getLogger(__name__)


class AnimalIngress(BaseIngress):
    ingress_method: ClassVar[str] = "animal"

    def _ingress_reader(self):
        def stage_index_from_file_path(file_path: FilePath) -> int:
            return int(file_path.stem.split(".")[0])

        for animal_dir in self.dataset_directory_path.iterdir():
            animal_id = int(animal_dir.stem) if animal_dir.stem.isdigit() else animal_dir.stem

            for trial_data_filename in animal_dir.glob(f"*{self.kinematic_data_file_extension}"):
                trial_data_filename = Path(trial_data_filename)
                stage_index = stage_index_from_file_path(trial_data_filename)

                trial_id = _define_trial_id(animal_id, stage_index)
                trial_id = int(trial_id) if trial_id.isdigit() else trial_id

                if trial_id not in self.metadata.index:
                    continue

                self._trial_id_to_trial_class_name[trial_id] = self.experiment_class.stage_index_to_trial_class_name[
                    stage_index
                ]
                self._trial_id_to_keyword_arguments[trial_id] = {
                    "label": trial_id,
                    "animal_id": animal_id,
                    "stage": stage_index,
                    "coordinate_data_path": trial_data_filename,
                }
                for plugin_info in self._trial_wise_plugins:
                    plugin_data_files = tuple(animal_dir.glob(f"{stage_index}.{plugin_info['code_key']}*"))
                    if len(plugin_data_files) > 1:
                        msg = f"Plugin {plugin_info['human_readable_index']}: Only one file per trial"
                        raise ValueError(msg)

                    self._trial_id_to_keyword_arguments[trial_id][plugin_info["bikipy_trial_key"]] = plugin_info[
                        "file_path_to_value"
                    ](plugin_data_files[0], trial_id, self)

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


def _animal_ids(project_root_directory: DirectoryPath):
    animal_ids = set()
    for trial_set_dir in get_dataset_directory_path(project_root_directory).iterdir():
        if trial_set_dir.is_dir():
            animal_ids.add(trial_set_dir.name.split("-")[0])
    return animal_ids


def _define_trial_id(animal_id: int | str, stage_index: int | str):
    return f"{animal_id}_{stage_index}"
