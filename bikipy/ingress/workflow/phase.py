from logging import getLogger
from typing import ClassVar

from pydantic import DirectoryPath

from bikipy.core.typing import Label
from bikipy.ingress.workflow.base import BaseIngressWorkflow

logger = getLogger(__name__)


class PhaseIngressWorkflow(BaseIngressWorkflow):
    ingress_method: ClassVar[str] = "phase"

    _coordinate_file_index_delimiter = "-"

    def _dataset_reader(self) -> None:
        """
        TODO: Consider code cleanup,
        :return:
        """
        if self.experiment_class.has_stages and self.only_one_instance_of_trial_class:
            trial_class_include = {cls.__name__: False for cls in self.experiment_class.trial_sequence}

        for phase_dir in self.dataset_directory.iterdir():
            phase_id = self._get_id_from_path_stem(phase_dir)

            for framewise_coordinates_path in self._coordinate_files_in_directory(phase_dir):
                phase_designated_trial_id = self._get_id_from_path_stem(framewise_coordinates_path)
                trial_id = _define_trial_id(phase_id, phase_designated_trial_id)

                if self._to_skip_trial_id(trial_id):
                    continue

                if self.experiment_class.has_stages:
                    stage_index = self.metadata.loc[trial_id, "Stage"]
                    trial_class = self._trial_class_from_stage_index(stage_index)
                    if self.only_one_instance_of_trial_class and trial_class_include[trial_class]:
                        continue

                trial_number = int(trial_id.split("_")[1])

                self._trial_id_to_keyword_arguments[trial_id] = {
                    "label": trial_id,
                    "animal_id": self.metadata.loc[trial_id, "Animal"],
                    "framewise_coordinates_path": framewise_coordinates_path,
                    **self.trialwise_plugins_for_trial_id(trial_number, phase_dir),
                    **self._trial_id_to_keyword_arguments[trial_id],
                }

                if self.experiment_class.has_stages:
                    self._trial_id_to_trial_class_name[trial_id] = trial_class

                    if self.only_one_instance_of_trial_class:
                        trial_class_include[trial_class] = True
                        if all(iter(trial_class_include.values())):
                            break

                elif self.only_one_instance_of_trial_class:
                    break

    def trialwise_plugins_for_trial_id(self, trial_id: Label, trial_directory: DirectoryPath):
        return self._trialwise_plugins_for_trial_id(trial_id, trial_directory, "{trial_id}-{plugin_code_key}*")


def _define_trial_id(phase_id: Label, trial_index: Label):
    return f"{phase_id}_{trial_index}"
