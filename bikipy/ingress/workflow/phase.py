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
        class_name = None
        observed_classes_to_trial_id = {}

        for phase_dir in self.dataset_directory.iterdir():
            phase_id = self._get_id_from_path_stem(phase_dir)

            for framewise_coordinates_path in self._coordinate_files_in_directory(phase_dir):
                phase_designated_trial_id = self._get_id_from_path_stem(framewise_coordinates_path)
                trial_id = _define_trial_id(phase_id, phase_designated_trial_id)

                if self._to_skip_trial_id(trial_id):
                    continue

                if self.experiment_class.has_stages:
                    stage_index = self.metadata.loc[trial_id, "Stage"]
                    class_name = self._trial_class_from_stage_index(stage_index).__name__
                    self._trial_id_to_trial_class_name[trial_id] = class_name

                trial_number = int(trial_id.split("_")[1])

                self._trial_id_to_keyword_arguments[trial_id] = {
                    "label": trial_id,
                    "animal_id": self.metadata.loc[trial_id, "Animal"],
                    "framewise_coordinates_path": framewise_coordinates_path,
                    **self.trialwise_plugins_for_trial_id(trial_number, phase_dir),
                    **self._trial_id_to_keyword_arguments[trial_id],
                }

                if self.only_one_instance_of_trial_class:
                    observed_classes_to_trial_id[class_name] = trial_id
                    if (
                        not self.experiment_class.has_stages
                        or frozenset(observed_classes_to_trial_id) == self.experiment_class.trial_classes
                    ):
                        for trial_id in tuple(self._trial_id_to_keyword_arguments):
                            if trial_id not in observed_classes_to_trial_id.values():
                                del self._trial_id_to_keyword_arguments[trial_id]
                        return

    def trialwise_plugins_for_trial_id(self, trial_id: Label, trial_directory: DirectoryPath):
        return self._trialwise_plugins_for_trial_id(trial_id, trial_directory, "{trial_id}-{plugin_code_key}*")


def _define_trial_id(phase_id: Label, trial_index: Label):
    return f"{phase_id}_{trial_index}"
