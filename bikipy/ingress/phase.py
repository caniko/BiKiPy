from logging import getLogger
from typing import ClassVar

from pydantic import PositiveInt, DirectoryPath

from bikipy.ingress.core import BaseIngress


logger = getLogger(__name__)


class PhaseIngress(BaseIngress):
    ingress_method: ClassVar[str] = "phase"

    def _dataset_reader(self) -> None:
        for phase_dir in self.dataset_directory_path.iterdir():
            phase_id = self._get_id_from_path_stem(phase_dir)

            for trial_kinematic_data_file_path in phase_dir.glob(f"*{self.kinematic_data_file_extension}"):
                phase_designated_trial_id = self._get_id_from_path_stem(trial_kinematic_data_file_path)
                trial_id = _define_trial_id(phase_id, phase_designated_trial_id)
                trial_number = int(trial_id.split("_")[1])

                if trial_id not in self.metadata.index:
                    continue

                self._trial_id_to_keyword_arguments[trial_id] = {
                    "label": trial_id,
                    "animal_id": self.metadata.loc[trial_id, "Animal"],
                    "coordinate_data_path": trial_kinematic_data_file_path,
                    **self.trialwise_plugins_for_trial_id(trial_number, phase_dir),
                }

                if self.experiment_class.has_stages:
                    stage_index = self.metadata.loc[trial_id, "Stage"]
                    self._trial_id_to_keyword_arguments[trial_id]["stage"] = stage_index
                    self._trial_id_to_trial_class_name[trial_id] = self._trial_class_from_stage_index(stage_index)

    def trialwise_plugins_for_trial_id(self, trial_id: str | PositiveInt, trial_directory: DirectoryPath):
        return self._trialwise_plugins_for_trial_id(trial_id, trial_directory, "{trial_id}-{plugin_code_key}*")


def _define_trial_id(phase_id: str | PositiveInt, stage_index: str | PositiveInt):
    return f"{phase_id}_{stage_index}"
