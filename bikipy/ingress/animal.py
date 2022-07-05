from logging import getLogger
from typing import ClassVar

from pydantic import PositiveInt, DirectoryPath

from bikipy.ingress.core import BaseIngress

logger = getLogger(__name__)


class AnimalIngress(BaseIngress):
    ingress_method: ClassVar[str] = "animal"

    def _dataset_reader(self) -> None:
        for animal_dir in self.dataset_directory_path.iterdir():
            animal_id = self._get_id_from_path_stem(animal_dir)

            for trial_kinematic_data_file_path in animal_dir.glob(f"*{self.kinematic_data_file_extension}"):
                stage_index = self._get_id_from_path_stem(trial_kinematic_data_file_path)

                trial_id = _define_trial_id(animal_id, stage_index)

                if trial_id not in self.metadata.index:
                    continue

                plugin_data = {}
                for plugin_info in self._trial_wise_plugins:
                    plugin_data_files = tuple(animal_dir.glob(f"{trial_id}.{plugin_info['code_key']}*"))
                    if len(plugin_data_files) > 1:
                        msg = f"Plugin {plugin_info['human_readable_index']}: Only one file per trial"
                        raise ValueError(msg)

                    data_object = plugin_info["file_path_to_value"](plugin_data_files[0], trial_id, self)
                    plugin_data[plugin_info["bikipy_trial_key"] or data_object.bikipy_trial_key] = data_object

                self._trial_id_to_trial_class_name[trial_id] = self._trial_class_from_stage_index(trial_id)
                self._trial_id_to_keyword_arguments[trial_id] = {
                    "label": trial_id,
                    "animal_id": animal_id,
                    "stage": stage_index,
                    "coordinate_data_path": trial_kinematic_data_file_path,
                    **self.trialwise_plugins_for_trial_id(trial_id, animal_dir),
                    **plugin_data,
                }

    def trialwise_plugins_for_trial_id(self, trial_id: str | PositiveInt, trial_directory: DirectoryPath):
        return self._trialwise_plugins_for_trial_id(trial_id, trial_directory, "{trial_id}.{plugin_code_key}*")

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
            msg = f"Animal column, Animal, is not defined in the metadata sheet. Defined columns:\n{self.metadata.columns}"
            raise KeyError(msg)

        if metadata_animal_id_column_set.issubset(animal_ids):
            msg = (
                "The animal ID sets in the metadata and the trial_set directory names do not match:\n"
                f"- metadata: {sorted(metadata_animal_id_column_set)}\n- trial_sets: {sorted(animal_ids)}"
            )
            raise ValueError(msg)


def _define_trial_id(animal_id: str | PositiveInt, stage_index: str | PositiveInt):
    return f"{animal_id}_{stage_index}"
